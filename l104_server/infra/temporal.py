"""
Temporal Memory Decay Engine

Extracted from engines_infra.py during EVO_78 refactoring.
Contains: TemporalMemoryDecayEngine - age-weighted memory decay.
"""

import time
import threading
import math
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497


class TemporalMemoryDecayEngine:
    """
    Applies temporal decay to memory entries.
    Older memories decay exponentially with PHI-weighted half-life.
    """

    PHI = PHI
    GOD_CODE = GOD_CODE
    VOID_CONSTANT = VOID_CONSTANT

    def __init__(self, half_life_seconds: float = 3600.0):
        """Initialize temporal decay engine."""
        self._half_life = half_life_seconds
        self._decay_constant = math.log(2) / half_life_seconds
        self._memories: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._decay_stats = {
            'total_decayed': 0,
            'total_removed': 0,
            'last_decay_run': None,
        }

    def register(self, memory_id: str, initial_weight: float = 1.0,
                 metadata: Optional[Dict] = None):
        """Register a memory entry for decay tracking."""
        with self._lock:
            self._memories[memory_id] = {
                'weight': initial_weight,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'metadata': metadata or {},
            }

    def access(self, memory_id: str) -> Optional[float]:
        """Access a memory, updating its weight and last accessed time."""
        with self._lock:
            if memory_id not in self._memories:
                return None
            try:
                memory = self._memories[memory_id]
                memory['last_accessed'] = time.time()
                memory['access_count'] += 1
                memory['weight'] *= (1.0 + 0.1 * self.PHI)
                return memory['weight']
            except Exception:
                return None

    def apply_decay(self) -> Dict[str, float]:
        """Apply temporal decay to all memories."""
        now = time.time()
        decayed = {}
        with self._lock:
            try:
                to_remove = []
                for memory_id, memory in self._memories.items():
                    age = now - memory['created_at']
                    raw_decay = math.exp(-self._decay_constant * age)
                    access_factor = 1.0 + (memory['access_count'] * 0.1 * (self.PHI - 1))
                    adjusted_decay = raw_decay ** (1.0 / access_factor)
                    memory['weight'] = adjusted_decay
                    decayed[memory_id] = adjusted_decay
                    if adjusted_decay < 0.01:
                        to_remove.append(memory_id)
                for memory_id in to_remove:
                    del self._memories[memory_id]
                    self._decay_stats['total_removed'] += 1
                self._decay_stats['total_decayed'] += len(decayed)
                self._decay_stats['last_decay_run'] = now
            except Exception:
                pass
        return decayed

    def clear(self):
        """Clear all memories and reset state."""
        with self._lock:
            try:
                self._memories.clear()
                self._decay_stats = {
                    'total_decayed': 0,
                    'total_removed': 0,
                    'last_decay_run': None,
                }
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get decay engine statistics."""
        with self._lock:
            try:
                weights = [m['weight'] for m in self._memories.values()]
                return {
                    'memory_count': len(self._memories),
                    'half_life_seconds': self._half_life,
                    'average_weight': sum(weights) / len(weights) if weights else 0.0,
                    'stats': dict(self._decay_stats),
                }
            except Exception:
                return {'memory_count': 0, 'error': True}


_temporal_decay = None

def get_temporal_decay() -> TemporalMemoryDecayEngine:
    """Get singleton TemporalMemoryDecayEngine instance."""
    global _temporal_decay
    if _temporal_decay is None:
        _temporal_decay = TemporalMemoryDecayEngine()
    return _temporal_decay


__all__ = ['TemporalMemoryDecayEngine', 'get_temporal_decay']
