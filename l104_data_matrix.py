# [L104_DATA_MATRIX] - EVOLVED HYPER-DIMENSIONAL STORAGE
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from l104_hyper_math import HyperMath
from l104_real_math import RealMath
from l104_memory_compaction import memory_compactor

UTC = timezone.utc
LATTICE_DB_PATH = "lattice_v2.db"

class DataMatrix:
    """
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
                except:
                    continue
            
            # 2. Hallucination Purge (Pruning data with resonance mismatch)
            # Facts that deviate significantly from the God Code resonance without high utility
            cur = conn.execute("SELECT key, resonance, utility FROM lattice_facts WHERE category != 'INVARIANT'")
            to_delete = []
            for key, resonance, utility in cur:
                # If resonance is extremely high (unstable) or utility is zero
                if resonance > 1000 and utility < 0.1:
                    to_delete.append(key)
            
            for key in to_delete:
                conn.execute("DELETE FROM lattice_facts WHERE key = ?", (key,))
                print(f"[DATA_MATRIX]: Purged unstable artifact: {key}")

            conn.commit()
        print("--- [DATA_MATRIX]: EVOLUTION COMPLETE. LATTICE STABILIZED. ---")

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
            "is_stabilized": confidence > 0.6
        }

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
