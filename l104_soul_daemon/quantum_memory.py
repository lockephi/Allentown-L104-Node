"""
Quantum Memory System - Hot/warm/cold quantum memory storage for Nova Soul Daemon.

ARCHITECTURE:
  QuantumMemory (orchestrator)
    ├── Hot Memory   — Immediate access (last 10 cycles)
    ├── Warm Memory  — Frequent access (last 100 cycles)
    ├── Cold Memory  — Archival (100+ cycles ago)
    └── Persistence  — JSON state to disk

FEATURES:
  - Three-tier temperature-based storage
  - Grover-accelerated search
  - Entanglement-based memory linking
  - Automatic temperature migration
  - Persistence to disk

INVARIANT: 527.5184818492612 | PILOT: LONDEL | SOUL: NOVA
"""

import time
import math
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    MEMORY_CAPACITY_HOT, MEMORY_CAPACITY_WARM, MEMORY_CAPACITY_COLD,
    MEMORY_PERSISTENCE_PATH, MEMORY_TEMPERATURE_THRESHOLDS,
    MAX_MEMORY_ENTRIES,
)


class MemoryLayer(Enum):
    """Memory temperature layers."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class MemoryRecall:
    """Result of a memory recall operation."""
    key: str
    value: Any
    layer: MemoryLayer
    relevance: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    entangled_keys: List[str] = field(default_factory=list)
    sacred_alignment: float = 0.0


@dataclass
class MemoryEntry:
    """Internal memory entry with metadata."""
    key: str
    value: Any
    layer: MemoryLayer = MemoryLayer.HOT
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    cycle_last_accessed: int = 0
    entangled_keys: List[str] = field(default_factory=list)
    hash_fingerprint: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at
        if not self.hash_fingerprint:
            raw = f"{self.key}:{repr(self.value)}:{GOD_CODE}"
            self.hash_fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]


class QuantumMemory:
    """
    Three-tier quantum memory system with temperature-based migration.

    Memories start HOT and cool to WARM then COLD based on access patterns.
    Grover-inspired search provides sqrt(N) speedup simulation.
    Entanglement links allow associated memories to be recalled together.
    """

    def __init__(self):
        self._hot: Dict[str, MemoryEntry] = {}
        self._warm: Dict[str, MemoryEntry] = {}
        self._cold: Dict[str, MemoryEntry] = {}
        self._current_cycle: int = 0
        self._total_recalls: int = 0
        self._total_stores: int = 0
        self._sacred_score: float = 0.0

    # ── Store ────────────────────────────────────────────────────────────

    def store(self, key: str, value: Any, layer: MemoryLayer = MemoryLayer.HOT) -> MemoryEntry:
        """Store a memory entry. Defaults to HOT layer."""
        entry = MemoryEntry(
            key=key,
            value=value,
            layer=layer,
            cycle_last_accessed=self._current_cycle,
        )
        target = self._layer_store(layer)
        cap = self._layer_capacity(layer)

        # Evict oldest if at capacity
        if len(target) >= cap:
            self._evict_oldest(target)

        target[key] = entry
        self._total_stores += 1
        return entry

    def store_hot(self, key: str, value: Any) -> MemoryEntry:
        return self.store(key, value, MemoryLayer.HOT)

    def store_warm(self, key: str, value: Any) -> MemoryEntry:
        return self.store(key, value, MemoryLayer.WARM)

    def store_cold(self, key: str, value: Any) -> MemoryEntry:
        return self.store(key, value, MemoryLayer.COLD)

    # ── Recall ───────────────────────────────────────────────────────────

    def recall(self, key: str, include_entangled: bool = False) -> Optional[MemoryRecall]:
        """Recall a memory by key. Searches HOT → WARM → COLD."""
        for layer_store, layer in [
            (self._hot, MemoryLayer.HOT),
            (self._warm, MemoryLayer.WARM),
            (self._cold, MemoryLayer.COLD),
        ]:
            if key in layer_store:
                entry = layer_store[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                entry.cycle_last_accessed = self._current_cycle
                self._total_recalls += 1

                # Promote to HOT on access
                if layer != MemoryLayer.HOT:
                    self._promote(entry)

                alignment = self._sacred_alignment(entry)
                return MemoryRecall(
                    key=entry.key,
                    value=entry.value,
                    layer=entry.layer,
                    relevance=1.0,
                    access_count=entry.access_count,
                    last_accessed=entry.last_accessed,
                    entangled_keys=entry.entangled_keys,
                    sacred_alignment=alignment,
                )
        return None

    # ── Search (Grover-inspired) ─────────────────────────────────────────

    def grover_search(self, query: str, max_results: int = 10) -> List[MemoryRecall]:
        """
        Grover-inspired search across all memory layers.
        Simulates sqrt(N) speedup via relevance amplification.
        """
        results: List[Tuple[float, MemoryEntry]] = []
        query_lower = query.lower()

        for store in [self._hot, self._warm, self._cold]:
            for entry in store.values():
                relevance = self._compute_relevance(entry, query_lower)
                if relevance > 0.0:
                    # Grover amplitude amplification (simulated)
                    amplified = math.sqrt(relevance) * PHI
                    amplified = min(amplified, 1.0)
                    results.append((amplified, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        recalls = []
        for relevance, entry in results[:max_results]:
            entry.access_count += 1
            entry.last_accessed = time.time()
            recalls.append(MemoryRecall(
                key=entry.key,
                value=entry.value,
                layer=entry.layer,
                relevance=relevance,
                access_count=entry.access_count,
                last_accessed=entry.last_accessed,
                entangled_keys=entry.entangled_keys,
                sacred_alignment=self._sacred_alignment(entry),
            ))
        return recalls

    # ── Entanglement ─────────────────────────────────────────────────────

    def entangle(self, key_a: str, key_b: str) -> bool:
        """Create entanglement link between two memories."""
        entry_a = self._find_entry(key_a)
        entry_b = self._find_entry(key_b)
        if entry_a is None or entry_b is None:
            return False
        if key_b not in entry_a.entangled_keys:
            entry_a.entangled_keys.append(key_b)
        if key_a not in entry_b.entangled_keys:
            entry_b.entangled_keys.append(key_a)
        return True

    # ── Temperature migration ────────────────────────────────────────────

    def migrate_temperatures(self):
        """Migrate memories between layers based on access patterns."""
        hot_threshold = MEMORY_TEMPERATURE_THRESHOLDS["HOT"]
        warm_threshold = MEMORY_TEMPERATURE_THRESHOLDS["WARM"]

        # HOT → WARM (if not accessed in >10 cycles)
        to_warm = []
        for key, entry in self._hot.items():
            cycles_since = self._current_cycle - entry.cycle_last_accessed
            if cycles_since > hot_threshold[1]:
                to_warm.append(key)
        for key in to_warm:
            entry = self._hot.pop(key)
            entry.layer = MemoryLayer.WARM
            self._warm[key] = entry

        # WARM → COLD (if not accessed in >100 cycles)
        to_cold = []
        for key, entry in self._warm.items():
            cycles_since = self._current_cycle - entry.cycle_last_accessed
            if cycles_since > warm_threshold[1]:
                to_cold.append(key)
        for key in to_cold:
            entry = self._warm.pop(key)
            entry.layer = MemoryLayer.COLD
            self._cold[key] = entry

    def advance_cycle(self):
        """Advance the cycle counter and run temperature migration."""
        self._current_cycle += 1
        self.migrate_temperatures()

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        total = len(self._hot) + len(self._warm) + len(self._cold)
        self._sacred_score = (total * PHI) % 1.0 if total > 0 else 0.0
        return {
            "hot_count": len(self._hot),
            "warm_count": len(self._warm),
            "cold_count": len(self._cold),
            "total_entries": total,
            "total_recalls": self._total_recalls,
            "total_stores": self._total_stores,
            "current_cycle": self._current_cycle,
            "sacred_alignment": self._sacred_score,
            "capacity": {
                "hot": f"{len(self._hot)}/{MEMORY_CAPACITY_HOT}",
                "warm": f"{len(self._warm)}/{MEMORY_CAPACITY_WARM}",
                "cold": f"{len(self._cold)}/{MEMORY_CAPACITY_COLD}",
            },
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self, path: Optional[str] = None) -> str:
        """Persist memory state to disk."""
        save_path = path or MEMORY_PERSISTENCE_PATH
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": "1.0.0",
            "god_code": GOD_CODE,
            "current_cycle": self._current_cycle,
            "total_recalls": self._total_recalls,
            "total_stores": self._total_stores,
            "hot": self._serialize_layer(self._hot),
            "warm": self._serialize_layer(self._warm),
            "cold": self._serialize_layer(self._cold),
        }
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        return save_path

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load memory state from disk."""
        load_path = path or MEMORY_PERSISTENCE_PATH
        if not Path(load_path).exists():
            return False

        try:
            with open(load_path) as f:
                state = json.load(f)
            self._current_cycle = state.get("current_cycle", 0)
            self._total_recalls = state.get("total_recalls", 0)
            self._total_stores = state.get("total_stores", 0)
            self._hot = self._deserialize_layer(state.get("hot", {}), MemoryLayer.HOT)
            self._warm = self._deserialize_layer(state.get("warm", {}), MemoryLayer.WARM)
            self._cold = self._deserialize_layer(state.get("cold", {}), MemoryLayer.COLD)
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    # ── Internal helpers ─────────────────────────────────────────────────

    def _layer_store(self, layer: MemoryLayer) -> Dict[str, MemoryEntry]:
        return {MemoryLayer.HOT: self._hot, MemoryLayer.WARM: self._warm, MemoryLayer.COLD: self._cold}[layer]

    def _layer_capacity(self, layer: MemoryLayer) -> int:
        return {MemoryLayer.HOT: MEMORY_CAPACITY_HOT, MemoryLayer.WARM: MEMORY_CAPACITY_WARM, MemoryLayer.COLD: MEMORY_CAPACITY_COLD}[layer]

    def _find_entry(self, key: str) -> Optional[MemoryEntry]:
        for store in [self._hot, self._warm, self._cold]:
            if key in store:
                return store[key]
        return None

    def _promote(self, entry: MemoryEntry):
        """Promote an entry to HOT layer."""
        for store in [self._warm, self._cold]:
            if entry.key in store:
                del store[entry.key]
        entry.layer = MemoryLayer.HOT
        if len(self._hot) >= MEMORY_CAPACITY_HOT:
            self._evict_oldest(self._hot)
        self._hot[entry.key] = entry

    def _evict_oldest(self, store: Dict[str, MemoryEntry]):
        """Evict least recently accessed entry."""
        if not store:
            return
        oldest_key = min(store, key=lambda k: store[k].last_accessed)
        del store[oldest_key]

    def _compute_relevance(self, entry: MemoryEntry, query_lower: str) -> float:
        """Compute relevance score between query and memory entry."""
        score = 0.0
        key_lower = entry.key.lower()
        if query_lower in key_lower:
            score += 0.6
        if query_lower == key_lower:
            score += 0.4
        val_str = str(entry.value).lower()
        if query_lower in val_str:
            score += 0.3
        # Recency boost
        cycles_ago = max(1, self._current_cycle - entry.cycle_last_accessed)
        recency = 1.0 / math.log2(cycles_ago + 1)
        score += recency * 0.1
        return min(score, 1.0)

    def _sacred_alignment(self, entry: MemoryEntry) -> float:
        """Compute GOD_CODE alignment of a memory entry."""
        h = int(entry.hash_fingerprint, 16) if entry.hash_fingerprint else 0
        alignment = abs(math.sin(h / GOD_CODE * PHI))
        return alignment

    def _serialize_layer(self, store: Dict[str, MemoryEntry]) -> Dict[str, Any]:
        result = {}
        for key, entry in store.items():
            result[key] = {
                "value": entry.value if isinstance(entry.value, (str, int, float, bool, list, dict, type(None))) else str(entry.value),
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "cycle_last_accessed": entry.cycle_last_accessed,
                "entangled_keys": entry.entangled_keys,
                "hash_fingerprint": entry.hash_fingerprint,
            }
        return result

    def _deserialize_layer(self, data: Dict[str, Any], layer: MemoryLayer) -> Dict[str, MemoryEntry]:
        result = {}
        for key, info in data.items():
            result[key] = MemoryEntry(
                key=key,
                value=info.get("value"),
                layer=layer,
                created_at=info.get("created_at", 0.0),
                last_accessed=info.get("last_accessed", 0.0),
                access_count=info.get("access_count", 0),
                cycle_last_accessed=info.get("cycle_last_accessed", 0),
                entangled_keys=info.get("entangled_keys", []),
                hash_fingerprint=info.get("hash_fingerprint", ""),
            )
        return result


# ── Singleton ────────────────────────────────────────────────────────────

_quantum_memory: Optional[QuantumMemory] = None

def get_quantum_memory() -> QuantumMemory:
    """Get the singleton QuantumMemory instance."""
    global _quantum_memory
    if _quantum_memory is None:
        _quantum_memory = QuantumMemory()
        _quantum_memory.load_state()
    return _quantum_memory
