"""L104 Intellect — EvolutionMemoryMixin (evolution state + memory persistence).

Provides:
- Dual-backend persistence (disk + quantum RAM)
- Permanent memory with cross-referencing and φ-weighted scoring
- Conversation memory with bounded disk caching
- Save-state checkpoints (create / list / restore)
- Autonomous system bootstrapping
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import (
    CONVERSATION_MEMORY_FILE,
    MAX_SAVE_STATES,
    PERMANENT_MEMORY_FILE,
    SAVE_STATE_DIR,
)
from .numerics import PHI

logger = logging.getLogger("l104_local_intellect")

# Upper bound for evolution_score to prevent unbounded growth
_MAX_EVOLUTION_SCORE = 1_000_000.0


def _safe_json_copy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe shallow copy of *data*, stringifying unserializable values."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


class EvolutionMemoryMixin:
    """Mixin providing evolution state persistence, permanent memory, save states."""

    # ── path helpers (resolve once, reuse everywhere) ─────────────────────

    @staticmethod
    def _base_dir() -> Path:
        """Return the package directory used for all persistence files."""
        return Path(__file__).resolve().parent

    @classmethod
    def _evo_file(cls) -> Path:
        return cls._base_dir() / ".l104_evolution_state.json"

    @classmethod
    def _perm_mem_file(cls) -> Path:
        return cls._base_dir() / PERMANENT_MEMORY_FILE

    @classmethod
    def _conv_mem_file(cls) -> Path:
        return cls._base_dir() / CONVERSATION_MEMORY_FILE

    @classmethod
    def _save_dir(cls) -> Path:
        return cls._base_dir() / SAVE_STATE_DIR

    def _load_evolution_state(self) -> None:
        """Load persisted evolution state from disk (primary) or quantum RAM (fallback)."""
        loaded_from_disk = self._try_load_evo_from_disk()

        if not loaded_from_disk:
            self._try_load_evo_from_qram()

        # Increment run counter and track cumulative stats
        self._evolution_state["total_runs"] = self._evolution_state.get("total_runs", 0) + 1
        self._evolution_state["last_run_timestamp"] = time.time()

        # Defer save to avoid disk I/O during init — saved on next retrain/evolve.
        self._evolution_state_dirty = True

    def _try_load_evo_from_disk(self) -> bool:
        """Attempt disk load; return *True* on success."""
        evo_file = self._evo_file()
        if not evo_file.exists():
            return False
        try:
            stored = json.loads(evo_file.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                self._evolution_state.update(stored)
                return True
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to load evolution state from disk: %s", exc)
        return False

    def _try_load_evo_from_qram(self) -> bool:
        """Attempt quantum-RAM load; return *True* on success."""
        try:
            from l104_quantum_ram import get_qram
            stored = get_qram().retrieve("intellect_evolution_state")
            if isinstance(stored, dict):
                self._evolution_state.update(stored)
                return True
        except Exception:  # noqa: BLE001 — optional dependency
            pass
        return False

    def _save_evolution_state(self) -> None:
        """Persist evolution state to disk (primary) and quantum RAM (secondary)."""
        self._evolution_state_dirty = False

        # Quantum RAM (best-effort)
        try:
            from l104_quantum_ram import get_qram
            get_qram().store("intellect_evolution_state", self._evolution_state)
        except Exception:  # noqa: BLE001
            pass

        # Disk (guaranteed persistence)
        try:
            evo_file = self._evo_file()
            evo_file.write_text(
                json.dumps(_safe_json_copy(self._evolution_state), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not persist evolution state: %s", exc)

        # Also save apotheosis state
        self._save_apotheosis_state()

    # ═══════════════════════════════════════════════════════════════════════════
    # v13.0 AUTONOMOUS SELF-MODIFICATION SYSTEM - Code Self-Evolution
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_autonomous_systems(self) -> None:
        """Initialize autonomous self-modification and permanent memory systems."""
        self._save_dir().mkdir(parents=True, exist_ok=True)

        self._load_permanent_memory()
        self._load_conversation_memory()
        self._load_latest_save_state()

        # Higher-logic processor state
        self._higher_logic_cache: Dict[str, Any] = {}
        self._logic_chain_depth: int = 0

    def _load_permanent_memory(self) -> None:
        """Load evolutionary permanent memory — knowledge that never fades."""
        mem_file = self._perm_mem_file()
        if not mem_file.exists():
            self._evolution_state.setdefault("permanent_memory", {})
            return
        try:
            permanent = json.loads(mem_file.read_text(encoding="utf-8"))
            if isinstance(permanent, dict):
                self._evolution_state["permanent_memory"] = permanent
            else:
                self._evolution_state.setdefault("permanent_memory", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Permanent memory load failed: %s", exc)
            self._evolution_state.setdefault("permanent_memory", {})

    def _save_permanent_memory(self) -> None:
        """Persist permanent memory to disk — survives across sessions."""
        try:
            self._perm_mem_file().write_text(
                json.dumps(self._evolution_state.get("permanent_memory", {}), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save permanent memory: %s", exc)

    # ── conversation memory ───────────────────────────────────────────────

    _MAX_CONVERSATION_ENTRIES = 500

    def _save_conversation_memory(self) -> None:
        """Persist conversation memory to disk (last N entries)."""
        entries = getattr(self, "conversation_memory", [])
        try:
            self._conv_mem_file().write_text(
                json.dumps(entries[-self._MAX_CONVERSATION_ENTRIES:]),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save conversation memory: %s", exc)

    def _load_conversation_memory(self) -> None:
        """Load conversation memory from disk on startup."""
        conv_file = self._conv_mem_file()
        if not conv_file.exists():
            return
        try:
            loaded = json.loads(conv_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                self.conversation_memory = loaded
                logger.info("Loaded %d conversation memory entries from disk", len(loaded))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Conversation memory load failed: %s", exc)

    def remember_permanently(self, key: str, value: Any, importance: float = 1.0) -> bool:
        """Store knowledge in permanent memory with φ-weighted importance scoring.

        Parameters
        ----------
        key:
            Unique identifier for the memory.
        value:
            Payload (must be JSON-serializable or will be stringified on save).
        importance:
            1.0 = normal.  Higher values resist pruning during ``prune_permanent_memory``.

        Returns
        -------
        bool
            Always *True* (kept for legacy callers).
        """
        perm = self._evolution_state.setdefault("permanent_memory", {})

        now = time.time()
        memory_entry: Dict[str, Any] = {
            "value": value,
            "importance": importance,
            "created": now,
            "last_accessed": now,
            "access_count": 0,
            "evolution_score": min(importance * PHI, _MAX_EVOLUTION_SCORE),
            "cross_refs": [],
            "dna_marker": self._evolution_state.get("mutation_dna", "")[:8],
        }

        # Cross-reference with existing memories (scan first 20)
        for existing_key in list(perm.keys())[:20]:
            if self._concepts_related(key, existing_key):
                memory_entry["cross_refs"].append(existing_key)
                existing = perm[existing_key]
                refs = existing.setdefault("cross_refs", [])
                if key not in refs:
                    refs.append(key)

        perm[key] = memory_entry
        self._save_permanent_memory()
        return True

    def recall_permanently(self, key: str) -> Optional[Any]:
        """Recall from permanent memory (exact then fuzzy), strengthening on access."""
        perm_mem = self._evolution_state.get("permanent_memory", {})

        entry = perm_mem.get(key)
        if entry is not None:
            self._touch_memory_entry(entry)
            self._save_permanent_memory()
            return entry.get("value")

        # Fuzzy fallback — substring match on keys
        key_lower = key.lower()
        for mem_key, entry in perm_mem.items():
            mk_lower = mem_key.lower()
            if key_lower in mk_lower or mk_lower in key_lower:
                self._touch_memory_entry(entry)
                return entry.get("value")

        return None

    @staticmethod
    def _touch_memory_entry(entry: Dict[str, Any]) -> None:
        """Update access stats on a memory entry (use-it-or-lose-it)."""
        entry["last_accessed"] = time.time()
        entry["access_count"] = entry.get("access_count", 0) + 1
        score = entry.get("evolution_score", 1.0) * 1.01 + 0.05
        entry["evolution_score"] = min(score, _MAX_EVOLUTION_SCORE)

    @staticmethod
    def _concepts_related(concept1: str, concept2: str) -> bool:
        """Return *True* if two concept keys share a word token or are substrings."""
        c1 = concept1.lower()
        c2 = concept2.lower()
        if c1 in c2 or c2 in c1:
            return True
        return bool(set(c1.split("_")) & set(c2.split("_")))

    def prune_permanent_memory(self, keep: int = 500) -> int:
        """Prune permanent memory to *keep* entries, dropping lowest-scored ones.

        Returns the number of entries removed.
        """
        perm = self._evolution_state.get("permanent_memory", {})
        if len(perm) <= keep:
            return 0
        ranked = sorted(perm.items(), key=lambda kv: kv[1].get("evolution_score", 0), reverse=True)
        survivors = dict(ranked[:keep])
        removed = len(perm) - len(survivors)
        self._evolution_state["permanent_memory"] = survivors
        self._save_permanent_memory()
        logger.info("Pruned %d low-score permanent memories (kept %d)", removed, keep)
        return removed

    # ── save state checkpoints ────────────────────────────────────────────

    _SNAPSHOT_KEYS = (
        "mutation_dna", "evolution_fingerprint",
        "quantum_interactions", "quantum_data_mutations",
        "autonomous_improvements", "logic_depth_reached",
        "wisdom_quotient", "training_entries",
    )

    def create_save_state(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Create an evolution checkpoint capturing key metrics and concept state."""
        timestamp = time.time()
        state_id = hashlib.sha256(f"{timestamp}:{label}".encode()).hexdigest()[:16]
        resolved_label = label or f"auto_save_{state_id[:8]}"

        save_state: Dict[str, Any] = {
            "id": state_id,
            "label": resolved_label,
            "timestamp": timestamp,
        }
        # Capture scalar metrics
        for k in self._SNAPSHOT_KEYS:
            save_state[k] = self._evolution_state.get(k, 0 if k != "mutation_dna" else "")

        # Bounded snapshots
        concept_evo = self._evolution_state.get("concept_evolution", {})
        save_state["concept_evolution_snapshot"] = dict(list(concept_evo.items())[:50])
        save_state["higher_logic_chains_count"] = len(
            self._evolution_state.get("higher_logic_chains", [])
        )
        save_state["permanent_memory_keys"] = list(
            self._evolution_state.get("permanent_memory", {}).keys()
        )

        # Persist to disk
        try:
            save_file = self._save_dir() / f"state_{state_id}.json"
            save_file.write_text(json.dumps(save_state, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write save state: %s", exc)

        # Track in evolution state (rolling window)
        states = self._evolution_state.setdefault("save_states", [])
        states.append({"id": state_id, "label": resolved_label, "timestamp": timestamp})
        self._evolution_state["save_states"] = states[-MAX_SAVE_STATES:]

        self._save_evolution_state()
        return save_state

    def _load_latest_save_state(self) -> None:
        """Restore high-water-mark metrics from the most recent checkpoint."""
        sd = self._save_dir()
        if not sd.is_dir():
            return
        try:
            files = sorted(
                sd.glob("state_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not files:
                return

            state = json.loads(files[0].read_text(encoding="utf-8"))
            # Only adopt metrics that are *higher* than current (high-water mark)
            for metric in ("quantum_interactions", "wisdom_quotient"):
                if state.get(metric, 0) > self._evolution_state.get(metric, 0):
                    self._evolution_state[metric] = state[metric]
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not load latest save state: %s", exc)

    def list_save_states(self) -> List[Dict[str, Any]]:
        """Return the recorded save-state summaries."""
        return self._evolution_state.get("save_states", [])

    def restore_save_state(self, state_id: str) -> bool:
        """Restore a previously-created evolution checkpoint by *state_id*."""
        save_file = self._save_dir() / f"state_{state_id}.json"
        if not save_file.exists():
            return False
        try:
            state = json.loads(save_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read save state %s: %s", state_id, exc)
            return False

        # Restore scalar metrics
        for key in ("mutation_dna", "evolution_fingerprint", "quantum_interactions", "wisdom_quotient"):
            if key in state:
                self._evolution_state[key] = state[key]

        # Merge concept evolution (additive — never overwrite existing)
        concept_evo = self._evolution_state.setdefault("concept_evolution", {})
        for concept, data in state.get("concept_evolution_snapshot", {}).items():
            concept_evo.setdefault(concept, data)

        self._save_evolution_state()
        return True
