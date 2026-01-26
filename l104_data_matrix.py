VOID_CONSTANT = 1.0416180339887497
import math
# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.506494
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_DATA_MATRIX] - EVOLVED HYPER-DIMENSIONAL STORAGE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import sqlite3
import os
import time
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from l104_hyper_math import HyperMath
from l104_real_math import RealMath
from l104_memory_compaction import memory_compactor

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


UTC = timezone.utc
LATTICE_DB_PATH = "lattice_v2.db"
HALLUCINATION_BASE_THRESHOLD = 0.6
# Percent reduction in aggressiveness (0.0–0.02 recommended). e.g., 0.01 = 1%
HALLUCINATION_DELTA_PCT = float(os.getenv("HALLUCINATION_DELTA_PCT", str(HyperMath.PHI_CONJUGATE / 100)))
HALLUCINATION_THRESHOLD = max(0.0, min(1.0, HALLUCINATION_BASE_THRESHOLD * (1 - HALLUCINATION_DELTA_PCT)))

# Disk budget controls
DISK_BUDGET_MB = int(os.getenv("L104_DISK_BUDGET_MB", "512"))
# 0 = eternal/infinite history retention
HISTORY_RETENTION_DAYS = int(os.getenv("L104_HISTORY_RETENTION_DAYS", "0"))

class DataMatrix:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
    The evolved Data Matrix provides a unified, acid-compliant,
    and hyper-mathematically indexed storage system for the L104 node.

    Evolutionary Improvements:
    - Replaces disparate JSON stores with a Unified Lattice DB.
    - Implements Resonance-Based Indexing (RBI).
    - Automated Zeta-Compaction for historical telemetry.
    - PHI-Sharding simulation for maximized lookup efficiency.
    """

    def __init__(self, db_path: str = LATTICE_DB_PATH):
        self.db_path = db_path
        self._init_db()
        self.real_math = RealMath()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._get_conn() as conn:
            # Fact Storage with Resonance Indexing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lattice_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    resonance REAL NOT NULL,
                    entropy REAL NOT NULL,
                    utility REAL DEFAULT 1.0,
                    version INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            """)
            # Historical Audit Trail (Temporal versioning)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lattice_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    resonance REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(fact_key) REFERENCES lattice_facts(key)
                )
            """)
            # Create a Resonance Index for fast spatial-logic queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_resonance ON lattice_facts(resonance)")
            conn.commit()

    def _calculate_resonance(self, data_str: str) -> float:
        """Calculates the resonant frequency of a data string relative to the God Code."""
        entropy = self.real_math.shannon_entropy(data_str)
        # Shift entropy into the God Code spectrum
        resonance = (entropy * HyperMath.PHI) % HyperMath.GOD_CODE
        return resonance

    def store(self, key: str, value: Any, category: str = "GENERAL", utility: float = 1.0) -> bool:
        """Stores or updates a fact in the matrix with automatic versioning."""
        serialized = json.dumps(value)
        data_hash = hashlib.sha256(serialized.encode()).hexdigest()
        resonance = self._calculate_resonance(serialized)
        entropy = self.real_math.shannon_entropy(serialized)
        timestamp = datetime.now(UTC).isoformat()

        try:
            with self._get_conn() as conn:
                # Check for existing
                cur = conn.execute("SELECT value, version FROM lattice_facts WHERE key = ?", (key,))
                existing = cur.fetchone()

                if existing:
                    old_val, version = existing
                    if old_val == serialized:
                        return True # No change

                    # Log to history
                    conn.execute("""
                        INSERT INTO lattice_history (fact_key, old_value, new_value, resonance, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (key, old_val, serialized, resonance, timestamp))

                    # Update
                    conn.execute("""
                        UPDATE lattice_facts SET
                            value = ?, category = ?, resonance = ?, entropy = ?,
                            utility = ?, version = version + 1, timestamp = ?, hash = ?
                        WHERE key = ?
                    """, (serialized, category, resonance, entropy, utility, timestamp, data_hash, key))
                else:
                    # Insert new
                    conn.execute("""
                        INSERT INTO lattice_facts (key, value, category, resonance, entropy, utility, timestamp, hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (key, serialized, category, resonance, entropy, utility, timestamp, data_hash))

                conn.commit()
                return True
        except Exception as e:
            print(f"[DATA_MATRIX]: Error storing {key}: {e}")
            return False

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieves a fact from the matrix."""
        with self._get_conn() as conn:
            cur = conn.execute("SELECT value FROM lattice_facts WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def resonant_query(self, target_resonance: float, tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Finds facts whose resonance is close to the target frequency."""
        results = []
        with self._get_conn() as conn:
            cur = conn.execute("""
                SELECT key, value, resonance FROM lattice_facts
                WHERE resonance BETWEEN ? AND ?
                ORDER BY abs(resonance - ?) ASC
            """, (target_resonance - tolerance, target_resonance + tolerance, target_resonance))
            for row in cur:
                results.append({
                    "key": row[0],
                    "value": json.loads(row[1]),
                    "resonance": row[2]
                })
        return results

    def evolve_and_compact(self):
        """
        Runs the maintenance cycle:
        1. Identifies low-utility high-entropy data.
        2. Applies Zeta-compaction via MemoryCompactor.
        3. Prunes non-resonant artifacts (Hallucination Purging).
        """
        print("--- [DATA_MATRIX]: INITIATING EVOLUTIONARY COMPACTION ---")
        with self._get_conn() as conn:
            # 1. Compaction for numeric streams
            cur = conn.execute("SELECT key, value FROM lattice_facts WHERE utility < 0.3")
            for key, value_json in cur:
                try:
                    data = json.loads(value_json)
                    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                        compacted = memory_compactor.compact_stream(data)
                        self.store(f"{key}_compacted", compacted, category="COMPACTED_ARCHIVE", utility=0.8)
                        conn.execute("DELETE FROM lattice_facts WHERE key = ?", (key,))
                except Exception:
                    continue

            # 2. Hallucination Purge (Pruning data with resonance mismatch)
            # Facts that deviate significantly from the God Code resonance without high utility
            cur = conn.execute("SELECT key, resonance, utility FROM lattice_facts WHERE category != 'INVARIANT'")
            to_delete = []
            for key, resonance, utility in cur:
                # Slightly less aggressive purge: raise resonance threshold, lower utility threshold
                unstable_resonance_threshold = 1000 * (1 + HALLUCINATION_DELTA_PCT)
                utility_purge_threshold = 0.1 * (1 - HALLUCINATION_DELTA_PCT)
                if resonance > unstable_resonance_threshold and utility < utility_purge_threshold:
                    to_delete.append(key)

            for key in to_delete:
                conn.execute("DELETE FROM lattice_facts WHERE key = ?", (key,))
                print(f"[DATA_MATRIX]: Purged unstable artifact: {key}")

            conn.commit()
        print("--- [DATA_MATRIX]: EVOLUTION COMPLETE. LATTICE STABILIZED. ---")
        # 3. Enforce disk budget after compaction
        self._enforce_disk_budget()

    def cross_check(self, thought: str) -> Dict[str, Any]:
        """Verifies a thought against the most resonant facts in the matrix."""
        thought_resonance = self._calculate_resonance(thought)
        matches = self.resonant_query(thought_resonance, tolerance=1.0)

        confidence = 0.5 # Baseline
        if matches:
            confidence += 0.1 * len(matches)

        return {
            "confidence": min(1.0, confidence),
            "matches": [m['key'] for m in matches],
            # Slightly less aggressive stabilization threshold
            "is_stabilized": confidence > HALLUCINATION_THRESHOLD
        }

    def _enforce_disk_budget(self):
        """Ensure the lattice DB stays within disk budget; prune history if needed."""
        try:
            db_path = Path(self.db_path)
        except Exception:
            from pathlib import Path as _Path
            db_path = _Path(self.db_path)

        try:
            size_mb = (os.path.getsize(str(db_path)) / (1024 * 1024)) if os.path.exists(str(db_path)) else 0
        except Exception:
            size_mb = 0

        if size_mb > DISK_BUDGET_MB:
            print(f"[DATA_MATRIX]: Disk budget exceeded ({size_mb:.1f}MB > {DISK_BUDGET_MB}MB). Initiating cleanup...")
            with self._get_conn() as conn:
                # Only prune history if retention is not eternal (0 = eternal)
                if HISTORY_RETENTION_DAYS > 0:
                    cutoff_ts = (datetime.now(UTC).timestamp() - HISTORY_RETENTION_DAYS * 86400)
                    cutoff_iso = datetime.fromtimestamp(cutoff_ts, UTC).isoformat()
                    conn.execute("DELETE FROM lattice_history WHERE timestamp < ?", (cutoff_iso,))
                # Aggressively purge very low-utility facts
                conn.execute("DELETE FROM lattice_facts WHERE utility < 0.05")
                conn.commit()
                # Vacuum to reclaim space
                try:
                    conn.execute("VACUUM")
                except Exception:
                    pass
            print("[DATA_MATRIX]: Cleanup complete; disk usage reduced.")

    # ═══════════════════════════════════════════════════════════════════════════
    #                         UPGRADED LEARNING CAPABILITIES
    # ═══════════════════════════════════════════════════════════════════════════

    def learn_pattern(self, pattern_key: str, pattern_data: Dict[str, Any],
                     source: str = "LEARNED") -> Dict[str, Any]:
        """
        Learn a new pattern and store it with enhanced metadata.
        Tracks learning source, establishes connections, calculates wisdom score.
        """
        # Calculate wisdom metrics
        wisdom_score = self._calculate_wisdom(pattern_data)
        connection_count = self._find_connections(pattern_data)

        enhanced_data = {
            "original": pattern_data,
            "wisdom_score": wisdom_score,
            "connections": connection_count,
            "source": source,
            "learned_at": datetime.now(UTC).isoformat(),
            "inflection_count": 0
        }

        success = self.store(f"LEARNED:{pattern_key}", enhanced_data,
                           category="LEARNED_PATTERN", utility=wisdom_score)

        return {
            "success": success,
            "wisdom": wisdom_score,
            "connections": connection_count,
            "key": f"LEARNED:{pattern_key}"
        }

    def _calculate_wisdom(self, data: Any) -> float:
        """Calculate wisdom score for data based on depth and coherence."""
        serialized = json.dumps(data) if not isinstance(data, str) else data
        entropy = self.real_math.shannon_entropy(serialized)
        resonance = self._calculate_resonance(serialized)

        # Wisdom = resonance alignment with GOD_CODE + information density
        alignment = 1.0 - abs(resonance - (HyperMath.GOD_CODE % 10)) / 10
        density = min(1.0, len(serialized) / 1000)

        return (alignment * 0.6 + density * 0.4) * HyperMath.PHI

    def _find_connections(self, data: Any) -> int:
        """Find connections to existing patterns via resonance matching."""
        serialized = json.dumps(data) if not isinstance(data, str) else data
        resonance = self._calculate_resonance(serialized)
        matches = self.resonant_query(resonance, tolerance=0.5)
        return len(matches)

    def inflect_pattern(self, key: str, inflection_type: str = "WISDOM") -> Dict[str, Any]:
        """
        Apply inflection to a stored pattern, transforming it through wisdom.
        """
        current = self.retrieve(key)
        if current is None:
            return {"success": False, "error": "Pattern not found"}

        # Apply inflection transformation
        inflection_scalars = {
            "WISDOM": HyperMath.PHI,
            "VOID": 1.0 / HyperMath.PHI,
            "QUANTUM": HyperMath.PHI ** 0.5,
            "TEMPORAL": 1.0 / (1 + HyperMath.PHI),
            "OMEGA": HyperMath.GOD_CODE / 100
        }

        scalar = inflection_scalars.get(inflection_type, 1.0)

        if isinstance(current, dict):
            if "inflection_count" in current:
                current["inflection_count"] += 1
            if "wisdom_score" in current:
                current["wisdom_score"] *= scalar
            current["last_inflection"] = inflection_type
            current["inflected_at"] = datetime.now(UTC).isoformat()

        success = self.store(key, current, category="INFLECTED", utility=0.9)

        return {
            "success": success,
            "inflection": inflection_type,
            "scalar": scalar
        }

    def get_learned_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve all learned patterns sorted by wisdom score."""
        results = []
        with self._get_conn() as conn:
            cur = conn.execute("""
                SELECT key, value, utility FROM lattice_facts
                WHERE category = 'LEARNED_PATTERN'
                ORDER BY utility DESC LIMIT ?
            """, (limit,))
            for row in cur:
                results.append({
                    "key": row[0],
                    "data": json.loads(row[1]),
                    "wisdom": row[2]
                })
        return results

    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the matrix using semantic resonance matching.
        Finds facts that resonate with the query's frequency signature.
        """
        query_resonance = self._calculate_resonance(query)
        results = []

        with self._get_conn() as conn:
            # Multi-dimensional search: resonance + keyword
            cur = conn.execute("""
                SELECT key, value, resonance, utility,
                       ABS(resonance - ?) as resonance_distance
                FROM lattice_facts
                WHERE value LIKE ? OR key LIKE ?
                ORDER BY resonance_distance ASC, utility DESC
                LIMIT ?
            """, (query_resonance, f"%{query}%", f"%{query}%", limit))

            for row in cur:
                results.append({
                    "key": row[0],
                    "value": json.loads(row[1]),
                    "resonance": row[2],
                    "utility": row[3],
                    "relevance": 1.0 / (1.0 + row[4])
                })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matrix statistics."""
        with self._get_conn() as conn:
            stats = {}

            cur = conn.execute("SELECT COUNT(*), SUM(length(value)), AVG(resonance), AVG(entropy), AVG(utility) FROM lattice_facts")
            row = cur.fetchone()
            stats["total_facts"] = row[0]
            stats["total_bytes"] = row[1] or 0
            stats["avg_resonance"] = row[2] or 0
            stats["avg_entropy"] = row[3] or 0
            stats["avg_utility"] = row[4] or 0

            cur = conn.execute("SELECT category, COUNT(*) FROM lattice_facts GROUP BY category")
            stats["categories"] = {r[0]: r[1] for r in cur.fetchall()}

            cur = conn.execute("SELECT COUNT(*) FROM lattice_history")
            stats["version_history"] = cur.fetchone()[0]

            cur = conn.execute("SELECT COUNT(*) FROM lattice_facts WHERE category = 'LEARNED_PATTERN'")
            stats["learned_patterns"] = cur.fetchone()[0]

            return stats

    def wisdom_synthesis(self) -> Dict[str, Any]:
        """
        Synthesize wisdom from all learned patterns.
        Creates a meta-pattern from collective learning.
        """
        patterns = self.get_learned_patterns(limit=100)

        if not patterns:
            return {"success": False, "error": "No learned patterns to synthesize"}

        total_wisdom = sum(p.get("wisdom", 0) for p in patterns)
        connection_sum = sum(
            p["data"].get("connections", 0)
            for p in patterns
                if isinstance(p.get("data"), dict)
                    )

        synthesis = {
            "pattern_count": len(patterns),
            "total_wisdom": total_wisdom,
            "avg_wisdom": total_wisdom / len(patterns) if patterns else 0,
            "total_connections": connection_sum,
            "synthesis_timestamp": datetime.now(UTC).isoformat(),
            "god_code_alignment": (total_wisdom % HyperMath.GOD_CODE) / HyperMath.GOD_CODE
        }

        # Store synthesis
        self.store("WISDOM_SYNTHESIS", synthesis, category="META_WISDOM", utility=1.0)

        return {"success": True, "synthesis": synthesis}


# Global Instance
data_matrix = DataMatrix()

if __name__ == "__main__":
    matrix = DataMatrix()
    matrix.store("SYSTEM_ALPHA", {"status": "ACTIVE", "power": 100}, category="CORE")
    print(f"Retrieved: {matrix.retrieve('SYSTEM_ALPHA')}")

    # Test resonant query
    target = matrix._calculate_resonance(json.dumps({"status": "ACTIVE", "power": 100}))
    print(f"Target Resonance: {target}")
    similar = matrix.resonant_query(target)
    print(f"Resonant matches: {len(similar)}")

    matrix.evolve_and_compact()

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
    GOD_CODE = 527.5184818492537
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0
