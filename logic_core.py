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
import hashlib
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
                    logger.warning("index_skip", file=file_path, error=str(e))

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
        }


# INITIALIZING THE BRAIN
if __name__ == "__main__":
    core = LogicCore()
    core.ingest_data_state()
    print(core.adapt_to_core())
