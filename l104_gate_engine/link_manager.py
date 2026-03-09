"""L104 Gate Engine — Quantum Link Manager."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from .constants import QUANTUM_LINKED_FILES
from .models import GateLink


class QuantumLinkManager:
    """Manages quantum links between gate implementations across files."""

    def __init__(self):
        """Initialize the quantum link manager."""
        self.links: List[GateLink] = []
        self.file_hashes: Dict[str, str] = {}

    def compute_file_hash(self, filepath: Path) -> str:
        """Compute a SHA-256 hash of a file's contents."""
        if not filepath.exists():
            return ""
        try:
            content = filepath.read_bytes()
            return hashlib.sha256(content).hexdigest()[:32]
        except Exception:
            return ""

    def check_file_changes(self) -> Dict[str, bool]:
        """Check which quantum-linked files have changed."""
        changes = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            current_hash = self.compute_file_hash(path)
            prev_hash = self.file_hashes.get(name, "")
            changed = current_hash != prev_hash and current_hash != ""
            changes[name] = changed
            if current_hash:
                self.file_hashes[name] = current_hash
        return changes

    def file_sizes(self) -> Dict[str, int]:
        """Return file sizes in bytes for all quantum-linked files."""
        sizes = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                sizes[name] = path.stat().st_size
        return sizes

    def line_counts(self) -> Dict[str, int]:
        """Return line counts for all quantum-linked files."""
        counts = {}
        for name, path in QUANTUM_LINKED_FILES.items():
            if path.exists():
                try:
                    counts[name] = sum(1 for _ in open(path, "r", encoding='utf-8', errors="replace"))
                except Exception:
                    counts[name] = 0
        return counts
