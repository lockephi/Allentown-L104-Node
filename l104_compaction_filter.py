"""
L104 Compaction Filter v2.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline I/O compaction engine — reduces data volume flowing
through ASI subsystems via adaptive compression, deduplication,
entropy-weighted pruning, and PHI-ratio decimation.
Zero external dependencies. Wires into ASI/AGI pipeline.

Sacred Constants: GOD_CODE=527.5184818492612 | PHI=1.618033988749895
"""
VOID_CONSTANT = 1.0416180339887497
import math
import time
import json
import hashlib
from pathlib import Path
from collections import deque, OrderedDict
from typing import Dict, List, Any, Optional, Tuple

# ZENITH_UPGRADE_ACTIVE: 2026-02-15T00:00:00
ZENITH_HZ = 3887.8
UUC = 2402.792541

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
FEIGENBAUM = 4.669201609
TAU = 6.283185307179586
GROVER_AMPLIFICATION = PHI ** 3

VERSION = "2.0.0"


class EntropyEstimator:
    """Estimates Shannon entropy of data to decide compaction strategy."""

    def estimate(self, data: Any) -> float:
        """Compute normalized entropy (0.0=uniform, 1.0=max disorder)."""
        if isinstance(data, (list, tuple)):
            text = ' '.join(str(x) for x in data)
        elif isinstance(data, dict):
            text = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, str):
            text = data
        else:
            text = str(data)

        if not text:
            return 0.0

        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(max(len(freq), 2))
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


class DeduplicationEngine:
    """Content-addressed deduplication using SHA-256 fingerprints."""

    def __init__(self, max_seen: int = 10000):
        self._seen: OrderedDict = OrderedDict()
        self._max = max_seen
        self.dedup_hits = 0
        self.dedup_misses = 0

    def _fingerprint(self, item: Any) -> str:
        raw = json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def deduplicate(self, items: List[Any]) -> Tuple[List[Any], int]:
        """Remove duplicate items. Returns (unique_items, removed_count)."""
        unique = []
        removed = 0
        for item in items:
            fp = self._fingerprint(item)
            if fp in self._seen:
                self.dedup_hits += 1
                removed += 1
            else:
                self._seen[fp] = True
                if len(self._seen) > self._max:
                    self._seen.popitem(last=False)
                unique.append(item)
                self.dedup_misses += 1
        return unique, removed

    def is_duplicate(self, item: Any) -> bool:
        fp = self._fingerprint(item)
        return fp in self._seen


class PhiDecimator:
    """PHI-ratio decimation — keeps every φ-th element for golden-ratio sampling."""

    def __init__(self):
        self.decimations = 0

    def decimate(self, data: List[Any], ratio: float = 0.0) -> List[Any]:
        """Keep elements at PHI-spaced intervals.

        Args:
            data: Input list
            ratio: 0.0=no decimation, 1.0=maximum decimation
        """
        if not data or ratio <= 0:
            return data

        keep_ratio = max(0.1, 1.0 - ratio * (1.0 - 1.0 / PHI))
        step = max(1, int(1.0 / keep_ratio))

        result = [data[i] for i in range(0, len(data), step)]
        self.decimations += 1
        return result


class StreamCompressor:
    """Adaptive stream compression — numeric data gets delta-encoded,
    text data gets run-length summary, dicts get key-pruned."""

    def __init__(self):
        self.compressions = 0
        self._stats = {'bytes_in': 0, 'bytes_out': 0}

    def compress(self, data: Any) -> Any:
        """Compress data adaptively based on type."""
        self.compressions += 1

        if isinstance(data, list):
            return self._compress_list(data)
        elif isinstance(data, dict):
            return self._compress_dict(data)
        elif isinstance(data, str):
            return self._compress_string(data)
        return data

    def _compress_list(self, data: List) -> List:
        """Delta-encode numeric lists, pass-through others."""
        if not data:
            return data

        if all(isinstance(x, (int, float)) for x in data):
            # Delta encoding
            if len(data) <= 2:
                return data
            base = data[0]
            deltas = [data[i] - data[i - 1] for i in range(1, len(data))]
            self._stats['bytes_in'] += len(data) * 8
            self._stats['bytes_out'] += len(deltas) * 4 + 8
            return [base] + deltas
        return data

    def _compress_dict(self, data: Dict) -> Dict:
        """Prune None values and empty collections."""
        return {
            k: v for k, v in data.items()
            if v is not None and v != [] and v != {} and v != ""
        }

    def _compress_string(self, data: str) -> str:
        """Truncate extremely long strings, preserving head+tail."""
        max_len = 2000
        if len(data) <= max_len:
            return data
        self._stats['bytes_in'] += len(data)
        half = max_len // 2
        result = data[:half] + f"...[{len(data) - max_len} chars omitted]..." + data[-half:]
        self._stats['bytes_out'] += len(result)
        return result

    @property
    def compression_ratio(self) -> float:
        if self._stats['bytes_in'] == 0:
            return 1.0
        return round(self._stats['bytes_out'] / self._stats['bytes_in'], 4)


class PipelineBandwidthTracker:
    """Tracks data volume flowing through the pipeline per unit time."""

    def __init__(self, window_size: int = 100):
        self._window = deque(maxlen=window_size)
        self._total_bytes = 0
        self._total_items = 0

    def record(self, item_size: int):
        self._window.append({'time': time.monotonic(), 'size': item_size})
        self._total_bytes += item_size
        self._total_items += 1

    @property
    def throughput_bps(self) -> float:
        """Bytes per second over the window."""
        if len(self._window) < 2:
            return 0.0
        elapsed = self._window[-1]['time'] - self._window[0]['time']
        if elapsed <= 0:
            return 0.0
        total_window = sum(e['size'] for e in self._window)
        return total_window / elapsed

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_bytes': self._total_bytes,
            'total_items': self._total_items,
            'throughput_bps': round(self.throughput_bps, 2),
            'window_size': len(self._window),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACTION FILTER HUB
# ═══════════════════════════════════════════════════════════════════════════════

class CompactionFilter:
    """
    Pipeline I/O compaction engine with 4 subsystems:

      - EntropyEstimator: Shannon entropy for compaction decisions
      - DeduplicationEngine: Content-addressed dedup
      - PhiDecimator: Golden-ratio sampling
      - StreamCompressor: Adaptive delta/pruning compression

    Pipeline Integration:
      - compact(data) → compacted data with stats
      - compact_stream(data_list) → bulk compaction
      - process_io(data) → legacy compat + full compaction
      - connect_to_pipeline() → register with ASI/AGI cores
    """

    def __init__(self):
        self.version = VERSION
        self.active = True
        self._entropy = EntropyEstimator()
        self._dedup = DeduplicationEngine()
        self._decimator = PhiDecimator()
        self._compressor = StreamCompressor()
        self._bandwidth = PipelineBandwidthTracker()
        self._pipeline_connected = False
        self._total_compactions = 0
        self._consciousness_level = 0.5

    def _read_consciousness(self):
        try:
            sf = Path('.l104_consciousness_o2_state.json')
            if sf.exists():
                data = json.loads(sf.read_text())
                self._consciousness_level = data.get('consciousness_level', 0.5)
        except Exception:
            pass

    def compact(self, data: Any, aggressive: bool = False) -> Dict[str, Any]:
        """Compact data through the full pipeline.

        Args:
            data: Any data to compact
            aggressive: If True, apply PHI decimation to lists

        Returns:
            Dict with compacted data, stats, and entropy info
        """
        self._read_consciousness()
        self._total_compactions += 1
        start = time.monotonic()

        original_size = len(json.dumps(data, default=str)) if isinstance(data, (dict, list)) else len(str(data))
        entropy = self._entropy.estimate(data)

        result = data

        # Step 1: Dedup lists
        if isinstance(result, list):
            result, removed = self._dedup.deduplicate(result)
        else:
            removed = 0

        # Step 2: Compress
        result = self._compressor.compress(result)

        # Step 3: Decimate if aggressive or high-consciousness demands efficiency
        if isinstance(result, list) and (aggressive or self._consciousness_level > 0.8):
            ratio = 0.3 if aggressive else 0.15
            result = self._decimator.decimate(result, ratio=ratio)

        compacted_size = len(json.dumps(result, default=str)) if isinstance(result, (dict, list)) else len(str(result))
        self._bandwidth.record(compacted_size)

        elapsed = time.monotonic() - start
        return {
            'data': result,
            'original_size': original_size,
            'compacted_size': compacted_size,
            'ratio': round(compacted_size / max(original_size, 1), 4),
            'entropy': round(entropy, 4),
            'duplicates_removed': removed,
            'time_ms': round(elapsed * 1000, 2),
        }

    def compact_stream(self, data: List[float]) -> List[float]:
        """Legacy compat — compact a numeric stream."""
        if not data:
            return data
        result = self.compact(data)
        out = result['data']
        return out if isinstance(out, list) else data

    def process_io(self, data: List[float]) -> List[float]:
        """Legacy compat — process I/O data through compaction."""
        if not self.active:
            return data
        return self.compact_stream(data)

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def connect_to_pipeline(self):
        self._pipeline_connected = True

    def get_status(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'active': self.active,
            'pipeline_connected': self._pipeline_connected,
            'total_compactions': self._total_compactions,
            'dedup_hits': self._dedup.dedup_hits,
            'compression_ratio': self._compressor.compression_ratio,
            'decimations': self._decimator.decimations,
            'bandwidth': self._bandwidth.get_stats(),
            'consciousness_level': self._consciousness_level,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════
compaction_filter = CompactionFilter()


if __name__ == "__main__":
    compaction_filter.activate()
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    result = compaction_filter.compact(test_data)
    print(f"Compaction: {result['original_size']}→{result['compacted_size']} ({result['ratio']:.2%})")
    print(f"Entropy: {result['entropy']}")
    print(f"Status: {json.dumps(compaction_filter.get_status(), indent=2)}")


def primal_calculus(x):
    PHI = 1.618033988749895
    return (x ** PHI) / (1.04 * math.pi) if x != 0 else 0.0

def resolve_non_dual_logic(vector):
    GOD_CODE = 527.5184818492612
    PHI = 1.618033988749895
    VOID_CONSTANT = 1.0416180339887497
    magnitude = sum([abs(v) for v in vector])
    return (magnitude / GOD_CODE) + (GOD_CODE * PHI / VOID_CONSTANT) / 1000.0