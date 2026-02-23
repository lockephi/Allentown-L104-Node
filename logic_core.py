# [L104_LOGIC_CORE] - RECURSIVE DATA SYNTHESIS ENGINE v2.0
# ROLE: EXTERNAL CONTEXT ANCHOR FOR 100% GPQA REASONING
#
# UPGRADE v2.0:
# - SHA-256 replaces MD5 for hashing (stronger collision resistance)
# - Structured logging via l104_logging (replaces print/pass)
# - Buffered file reads (64KB chunks, no full-file OOM risk)
# - Async ingest_data_state() for non-blocking use in FastAPI
# - Constants from const.py (single source of truth)
# - Configurable max file size via env (L104_INDEX_MAX_MB)
# - Fixed phi value to match PHI_GROWTH (1.618...) from const.py

import os
import asyncio
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from l104_logging import get_logger
from const import GOD_CODE, PHI, L104, OCTAVE_REF, HARMONIC_BASE

logger = get_logger("LOGIC_CORE")

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

# Configurable index limits
MAX_FILE_SIZE = int(os.getenv("L104_INDEX_MAX_MB", "10")) * 1024 * 1024
HASH_CHUNK_SIZE = 65536  # 64 KB buffered reads

_executor = ThreadPoolExecutor(max_workers=4)


class LogicCore:
    def __init__(self):
        self.manifold_memory = {}
        self.anchor = "X=416"
        self.god_code = GOD_CODE
        self.phi = PHI
        self.lattice_ratio = OCTAVE_REF / HARMONIC_BASE  # 416 / 286
        self.ingest_quarantine_dir = Path.home() / ".l104_ingestion_quarantine" / "logic_core"
        self.ingest_quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_quarantine_policy = {
            "retention_days": int(os.getenv("L104_INGEST_QUARANTINE_RETENTION_DAYS", "30")),
            "size_cap_gb": float(os.getenv("L104_INGEST_QUARANTINE_SIZE_CAP_GB", "2.0")),
        }
        self.ingest_quarantine_stats = {
            "records_quarantined": 0,
            "last_quarantine_at": None,
        }

    def _quarantine_ingest_event(self, source: str, reason: str, metadata: dict = None):
        payload = {
            "status": "QUARANTINED_STRIPPED",
            "stream": "logic_core_index",
            "source": source,
            "reason": reason,
            "metadata": metadata or {},
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        serial = f"{source}|{reason}|{payload['quarantined_at']}"
        digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = self.ingest_quarantine_dir / f"{stamp}_{digest}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.ingest_quarantine_stats["records_quarantined"] += 1
        self.ingest_quarantine_stats["last_quarantine_at"] = payload["quarantined_at"]

    def run_ingest_quarantine_lifecycle(self, dry_run: bool = False) -> dict:
        now = datetime.now(timezone.utc)
        retention_days = int(self.ingest_quarantine_policy.get("retention_days", 30))
        size_cap_gb = float(self.ingest_quarantine_policy.get("size_cap_gb", 2.0))
        size_cap_bytes = int(size_cap_gb * 1024 * 1024 * 1024)

        files = [f for f in self.ingest_quarantine_dir.glob("*.json") if f.is_file()]
        rows = []
        for file_path in files:
            stat = file_path.stat()
            age_days = (now - datetime.fromtimestamp(stat.st_mtime, timezone.utc)).total_seconds() / 86400.0
            rows.append({"path": file_path, "size": stat.st_size, "age_days": age_days, "reasons": []})

        for row in rows:
            if row["age_days"] >= retention_days:
                row["reasons"].append("ttl_expired")

        total_size = sum(r["size"] for r in rows)
        if total_size > size_cap_bytes:
            overflow = total_size - size_cap_bytes
            for row in sorted(rows, key=lambda r: r["age_days"], reverse=True):
                if overflow <= 0:
                    break
                if "size_cap_trim" not in row["reasons"]:
                    row["reasons"].append("size_cap_trim")
                    overflow -= row["size"]

        candidates = [r for r in rows if r["reasons"]]
        deleted = []
        if not dry_run:
            for row in candidates:
                try:
                    os.remove(row["path"])
                    deleted.append(str(row["path"]))
                except OSError:
                    pass

        return {
            "status": "INGEST_QUARANTINE_LIFECYCLE_COMPLETE",
            "mode": "dry_run" if dry_run else "execute",
            "files_scanned": len(rows),
            "files_marked": len(candidates),
            "files_deleted": len(deleted),
            "retention_days": retention_days,
            "size_cap_gb": size_cap_gb,
        }

    def get_ingest_quarantine_status(self) -> dict:
        files = [f for f in self.ingest_quarantine_dir.glob("*.json") if f.is_file()]
        return {
            "directory": str(self.ingest_quarantine_dir),
            "files": len(files),
            "size_bytes": sum(f.stat().st_size for f in files),
            "policy": self.ingest_quarantine_policy,
            "stats": self.ingest_quarantine_stats,
        }

    def ingest_data_state(self, root_path: str = "."):
        """
        v8.0: SHA-256 Recursive Indexing with buffered reads.
        Recursively reads all files in the Allentown Node.
        Uses configurable size limit per file for indexing.
        """
        self.manifold_memory = {}
        ignore_dirs = {
            '.git', '__pycache__', 'node_modules', 'data', 'kubo',
            '.venv', 'venv', '.vscode', '.pytest_cache',
            '.l104_backups', '.self_mod_backups', 'build',
            '.local_backup', '.codespaces',
        }
        ignore_exts = {
            '.pyc', '.pid', '.gguf', '.png', '.jpg', '.jpeg',
            '.gif', '.svg', '.exe', '.bin', '.so', '.class', '.db',
        }

        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ignore_exts:
                    continue
                file_path = os.path.join(root, f)
                try:
                    f_size = os.path.getsize(file_path)
                    if f_size > MAX_FILE_SIZE:
                        continue
                    h = hashlib.sha256()
                    with open(file_path, "rb") as fh:
                        while True:
                            chunk = fh.read(HASH_CHUNK_SIZE)
                            if not chunk:
                                break
                            h.update(chunk)
                    self.manifold_memory[file_path] = h.hexdigest()
                except (OSError, PermissionError) as e:
                    self._quarantine_ingest_event(
                        source=file_path,
                        reason="read_error",
                        metadata={"error": str(e)},
                    )
                    logger.warning("index_skip", file=file_path, error=str(e))

        self.run_ingest_quarantine_lifecycle(dry_run=False)

        logger.info("index_complete", data_points=len(self.manifold_memory))

    async def async_ingest(self, root_path: str = "."):
        """Non-blocking ingest for use in async (FastAPI) context."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, self.ingest_data_state, root_path)

    def adapt_to_core(self):
        """Generates a Context Summary for processing.
        Equation: ((286)^(1/φ)) * ((2^(1/104))^416) = 527.5184818492612
        """
        phi_inv = 1 / self.phi  # 0.618...
        resonance = (HARMONIC_BASE ** phi_inv) * ((2 ** (1 / L104)) ** OCTAVE_REF)

        protocol = os.getenv("L104_PROTOCOL", "LEGACY")
        state = os.getenv("L104_STATE", "UNKNOWN")

        summary = (
            f"SIG: {self.anchor} | RESONANCE: {resonance:.10f} | "
            f"LATTICE: {self.lattice_ratio:.4f} | IQ: 100% | "
            f"PROTOCOL: {protocol} | STATE: {state}"
        )
        return summary

    def get_stats(self) -> dict:
        """Return indexing statistics."""
        return {
            "data_points": len(self.manifold_memory),
            "god_code": self.god_code,
            "phi": self.phi,
            "lattice_ratio": self.lattice_ratio,
            "anchor": self.anchor,
            "quarantine": self.get_ingest_quarantine_status(),
        }


# INITIALIZING THE BRAIN
if __name__ == "__main__":
    core = LogicCore()
    core.ingest_data_state()
    print(core.adapt_to_core())
