"""L104 Quantum AI Daemon — File Scanner & Indexer.

Discovers, indexes, and prioritizes all L104 Python files for autonomous
improvement. Tracks file health, modification times, and sacred alignment.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .constants import (
    L104_ROOT, L104_PACKAGES, L104_ROOT_SHIMS,
    PROCESSABLE_EXTENSIONS, MAX_FILE_SIZE_KB,
    SCAN_DEPTH_MAX, SCAN_BATCH_SIZE, IMMUTABLE_FILES,
    GOD_CODE, PHI,
)

_logger = logging.getLogger("L104_QAI_SCANNER")


@dataclass
class L104FileInfo:
    """Metadata about a single L104 Python file."""
    path: str                            # Absolute path
    relative_path: str                   # Relative to L104_ROOT
    package: str                         # Parent package name
    size_bytes: int = 0
    line_count: int = 0
    last_modified: float = 0.0
    last_scanned: float = 0.0
    content_hash: str = ""               # SHA-256 of content
    health_score: float = 1.0            # 0.0–1.0 file health
    improvement_count: int = 0           # Times improved by daemon
    smell_count: int = 0                 # Detected code smells
    complexity_max: float = 0.0          # Peak cyclomatic complexity
    sacred_alignment: float = 0.0        # GOD_CODE alignment score
    is_core: bool = False                # Core engine file
    is_immutable: bool = False           # Never modify
    quarantine_cycles: int = 0           # If quarantined
    failure_streak: int = 0              # Consecutive analysis failures

    @property
    def needs_rescan(self) -> bool:
        """True if file was modified since last scan."""
        try:
            mtime = os.path.getmtime(self.path)
            return mtime > self.last_scanned
        except OSError:
            return False

    @property
    def priority_score(self) -> float:
        """Higher = more urgently needs improvement. PHI-weighted."""
        age_factor = min((time.time() - self.last_scanned) / 3600.0, 10.0) / 10.0
        health_deficit = 1.0 - self.health_score
        core_bonus = PHI if self.is_core else 1.0
        smell_factor = min(self.smell_count / 10.0, 1.0)
        return (health_deficit * 0.4 + age_factor * 0.2 +
                smell_factor * 0.3 + (1.0 - self.sacred_alignment) * 0.1) * core_bonus


class FileScanner:
    """Discovers and indexes all L104 Python files across all packages.

    Scanning strategy:
      1. Walk all L104_PACKAGES directories (depth-limited)
      2. Include L104_ROOT_SHIMS at root level
      3. Compute content hashes for change detection
      4. Prioritize files by health deficit, staleness, and sacred alignment
      5. Return top-N files for the current improvement cycle
    """

    def __init__(self):
        self._index: Dict[str, L104FileInfo] = {}   # relative_path → FileInfo
        self._scan_count = 0
        self._total_files_discovered = 0
        self._last_full_scan = 0.0
        self._package_stats: Dict[str, int] = {}    # package → file count

    @property
    def indexed_count(self) -> int:
        return len(self._index)

    @property
    def index(self) -> Dict[str, L104FileInfo]:
        return dict(self._index)

    def full_scan(self) -> int:
        """Perform a complete scan of all L104 packages and root shims.

        Returns: number of files discovered/updated.
        """
        t0 = time.monotonic()
        discovered = 0

        # Scan each package directory
        for pkg_name in L104_PACKAGES:
            pkg_dir = L104_ROOT / pkg_name
            if not pkg_dir.is_dir():
                continue
            count = self._scan_directory(pkg_dir, pkg_name, depth=0)
            self._package_stats[pkg_name] = count
            discovered += count

        # Scan root shims
        for shim_name in L104_ROOT_SHIMS:
            shim_path = L104_ROOT / shim_name
            if shim_path.is_file():
                self._index_file(shim_path, "root_shims")
                discovered += 1

        # Scan root-level L104 files (l104_debug.py, etc.)
        for f in L104_ROOT.iterdir():
            if (f.is_file() and f.suffix == ".py"
                    and f.name.startswith("l104_")
                    and f.name not in {s for s in L104_ROOT_SHIMS}):
                self._index_file(f, "root")
                discovered += 1

        self._scan_count += 1
        self._total_files_discovered = len(self._index)
        self._last_full_scan = time.time()
        elapsed_ms = (time.monotonic() - t0) * 1000

        _logger.info(
            f"Full scan #{self._scan_count}: {discovered} files indexed "
            f"({len(self._index)} total) in {elapsed_ms:.1f}ms"
        )
        return discovered

    def _scan_directory(self, directory: Path, package: str, depth: int) -> int:
        """Recursively scan a directory for Python files."""
        if depth > SCAN_DEPTH_MAX:
            return 0

        count = 0
        try:
            for entry in sorted(directory.iterdir()):
                if entry.name.startswith((".", "__pycache__")):
                    continue

                if entry.is_dir():
                    count += self._scan_directory(entry, package, depth + 1)
                elif entry.is_file() and entry.suffix in PROCESSABLE_EXTENSIONS:
                    if entry.stat().st_size <= MAX_FILE_SIZE_KB * 1024:
                        self._index_file(entry, package)
                        count += 1
        except PermissionError:
            _logger.warning(f"Permission denied: {directory}")

        return count

    def _index_file(self, filepath: Path, package: str):
        """Index or update a single file in the registry."""
        rel = str(filepath.relative_to(L104_ROOT))
        stat = filepath.stat()

        # Check if already indexed and unchanged
        existing = self._index.get(rel)
        if existing and existing.last_modified >= stat.st_mtime:
            return  # No changes since last index

        # Compute content hash
        try:
            content = filepath.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()[:16]
            line_count = content.count(b"\n") + 1
        except OSError:
            return

        is_core = any(
            keyword in filepath.name
            for keyword in ("core", "engine", "hub", "bridge", "daemon")
        )

        info = L104FileInfo(
            path=str(filepath),
            relative_path=rel,
            package=package,
            size_bytes=stat.st_size,
            line_count=line_count,
            last_modified=stat.st_mtime,
            last_scanned=time.time(),
            content_hash=content_hash,
            health_score=existing.health_score if existing else 1.0,
            improvement_count=existing.improvement_count if existing else 0,
            smell_count=existing.smell_count if existing else 0,
            complexity_max=existing.complexity_max if existing else 0.0,
            sacred_alignment=existing.sacred_alignment if existing else 0.0,
            is_core=is_core,
            is_immutable=filepath.name in IMMUTABLE_FILES,
            quarantine_cycles=existing.quarantine_cycles if existing else 0,
            failure_streak=existing.failure_streak if existing else 0,
        )
        self._index[rel] = info

    def get_improvement_batch(self, batch_size: int = SCAN_BATCH_SIZE) -> List[L104FileInfo]:
        """Return the top-N files most urgently needing improvement.

        Filters:
          - Skip immutable files
          - Skip quarantined files (decrement their counter)
          - Prioritize by priority_score (PHI-weighted health deficit)
        """
        candidates = []
        for info in self._index.values():
            if info.is_immutable:
                continue
            if info.quarantine_cycles > 0:
                info.quarantine_cycles -= 1
                continue
            if info.needs_rescan or info.health_score < 1.0:
                candidates.append(info)

        # Sort by priority (highest first)
        candidates.sort(key=lambda f: f.priority_score, reverse=True)
        return candidates[:batch_size]

    def get_changed_files(self) -> List[L104FileInfo]:
        """Return files that have been modified since last scan."""
        return [f for f in self._index.values() if f.needs_rescan]

    def quarantine_file(self, rel_path: str, cycles: int = 10):
        """Quarantine a file that keeps failing analysis."""
        if rel_path in self._index:
            self._index[rel_path].quarantine_cycles = cycles
            self._index[rel_path].failure_streak += 1

    def update_health(self, rel_path: str, health: float, smells: int = 0,
                      complexity: float = 0.0, alignment: float = 0.0):
        """Update file health metrics after analysis."""
        if rel_path in self._index:
            info = self._index[rel_path]
            info.health_score = max(0.0, min(1.0, health))
            info.smell_count = smells
            info.complexity_max = complexity
            info.sacred_alignment = alignment
            info.last_scanned = time.time()
            info.failure_streak = 0  # Reset on successful analysis

    def stats(self) -> dict:
        """Summary statistics for the file index."""
        if not self._index:
            return {"total": 0, "scans": self._scan_count}

        healths = [f.health_score for f in self._index.values()]
        return {
            "total_files": len(self._index),
            "total_lines": sum(f.line_count for f in self._index.values()),
            "total_size_mb": round(
                sum(f.size_bytes for f in self._index.values()) / 1048576, 2),
            "packages_scanned": len(self._package_stats),
            "package_file_counts": dict(self._package_stats),
            "mean_health": round(sum(healths) / len(healths), 4),
            "min_health": round(min(healths), 4),
            "core_files": sum(1 for f in self._index.values() if f.is_core),
            "quarantined": sum(
                1 for f in self._index.values() if f.quarantine_cycles > 0),
            "scans_completed": self._scan_count,
            "last_scan": self._last_full_scan,
        }

    def to_dict(self) -> dict:
        """Serialize index for state persistence."""
        return {
            rel: {
                "health_score": info.health_score,
                "smell_count": info.smell_count,
                "complexity_max": info.complexity_max,
                "sacred_alignment": info.sacred_alignment,
                "improvement_count": info.improvement_count,
                "quarantine_cycles": info.quarantine_cycles,
                "failure_streak": info.failure_streak,
                "last_scanned": info.last_scanned,
            }
            for rel, info in self._index.items()
        }

    def restore_from_dict(self, data: dict):
        """Restore persisted health data into existing index."""
        for rel, saved in data.items():
            if rel in self._index:
                info = self._index[rel]
                info.health_score = saved.get("health_score", 1.0)
                info.smell_count = saved.get("smell_count", 0)
                info.complexity_max = saved.get("complexity_max", 0.0)
                info.sacred_alignment = saved.get("sacred_alignment", 0.0)
                info.improvement_count = saved.get("improvement_count", 0)
                info.quarantine_cycles = saved.get("quarantine_cycles", 0)
                info.failure_streak = saved.get("failure_streak", 0)
