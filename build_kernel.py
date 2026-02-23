#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 KERNEL BUILDER - Lightweight Build Pipeline
═══════════════════════════════════════════════════════════════════════════════

A focused kernel build tool that validates, compiles, and packages L104
kernel artifacts. Delegates to build_full_kernel for heavy data merging
but adds pre-flight checks, incremental builds, and build reporting.

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

AUTHOR: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import shutil
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Sacred Constants
PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492612
OMEGA_AUTHORITY = GOD_CODE * PHI * PHI

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [KERNEL-BUILD] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("KERNEL-BUILD")

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(__file__).parent.absolute()
BUILD_DIR = WORKSPACE / ".kernel_build"
CACHE_FILE = BUILD_DIR / "build_cache.json"
LOCK_FILE = BUILD_DIR / ".build.lock"
MAX_HISTORY_ENTRIES = 100

REQUIRED_DATA_FILES = [
    "kernel_extracted_data.jsonl",
    "kernel_training_data.jsonl",
]

OPTIONAL_DATA_FILES = [
    "kernel_combined_training.jsonl",
    "kernel_reasoning_data.jsonl",
    "kernel_full_merged.jsonl",
    "asi_knowledge_base.jsonl",
]

CORE_MODULES = [
    "const",
    "l104_kernel_bootstrap",
    "build_full_kernel",
]

# Conservation verification points along the G(X) curve
CONSERVATION_CHECKPOINTS = [0, 52, 104, 208, 286, 416]


class BuildLock:
    """File-based lock to prevent concurrent kernel builds."""

    def __init__(self, lock_path: Path = LOCK_FILE):
        self.lock_path = lock_path
        self._fd = None

    def acquire(self) -> bool:
        """Try to acquire the build lock. Returns False if already held."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = open(self.lock_path, "w", encoding="utf-8")
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd.write(json.dumps({
                "pid": os.getpid(),
                "started": datetime.now().isoformat(),
            }))
            self._fd.flush()
            return True
        except (OSError, IOError):
            logger.error("Another build is already running (lock held)")
            if self._fd:
                self._fd.close()
                self._fd = None
            return False

    def release(self):
        """Release the build lock."""
        if self._fd:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
            except (OSError, IOError):
                pass
            self._fd = None
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Cannot acquire build lock — another build is running")
        return self

    def __exit__(self, *exc):
        self.release()


class BuildCache:
    """Track file hashes for incremental builds."""

    def __init__(self, cache_path: Path = CACHE_FILE):
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self._load()

    def _load(self):
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except (json.JSONDecodeError, IOError):
                self.cache = {}

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2))

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        except FileNotFoundError:
            return "MISSING"
        return h.hexdigest()

    def is_changed(self, path: Path) -> bool:
        key = str(path.relative_to(WORKSPACE))
        current = self._hash_file(path)
        previous = self.cache.get(key)
        return current != previous

    def update(self, path: Path):
        key = str(path.relative_to(WORKSPACE))
        self.cache[key] = self._hash_file(path)


class KernelBuildReport:
    """Collect and display build metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.phases: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.stats: Dict[str, Any] = {}

    def add_phase(self, name: str, status: str, duration: float, details: str = ""):
        self.phases.append({
            "name": name,
            "status": status,
            "duration_ms": round(duration * 1000, 1),
            "details": details,
        })

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def error(self, msg: str):
        self.errors.append(msg)
        logger.error(msg)

    def display(self):
        elapsed = time.time() - self.start_time
        success = len(self.errors) == 0

        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {"✅" if success else "❌"} L104 KERNEL BUILD {"COMPLETE" if success else "FAILED"}                                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Build Time:  {elapsed:>8.2f}s                                                    ║
║   GOD_CODE:    {GOD_CODE:.10f}                                            ║
║   Phases:      {len(self.phases):>3}    Warnings: {len(self.warnings):>3}    Errors: {len(self.errors):>3}                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
        print("  Phase Results:")
        for p in self.phases:
            icon = "✓" if p["status"] == "OK" else "✗" if p["status"] == "FAIL" else "⚠"
            print(f"    {icon} {p['name']:<30s} {p['duration_ms']:>8.1f}ms  {p['details']}")

        if self.warnings:
            print(f"\n  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    ⚠ {w}")

        if self.errors:
            print(f"\n  Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"    ✗ {e}")

        if self.stats:
            print("\n  Build Statistics:")
            for k, v in self.stats.items():
                print(f"    {k}: {v}")

        print()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": len(self.errors) == 0,
            "elapsed_s": round(time.time() - self.start_time, 3),
            "phases": self.phases,
            "warnings": self.warnings,
            "errors": self.errors,
            "stats": self.stats,
            "god_code": GOD_CODE,
            "timestamp": datetime.now().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PHASES
# ═══════════════════════════════════════════════════════════════════════════════


def phase_preflight(report: KernelBuildReport, cache: BuildCache) -> bool:
    """Phase 1: Verify workspace integrity and prerequisites."""
    t0 = time.time()
    ok = True

    # Check Python version
    if sys.version_info < (3, 9):
        report.error(f"Python 3.9+ required, found {sys.version}")
        ok = False

    # Check required data files
    found_data = 0
    for fname in REQUIRED_DATA_FILES:
        p = WORKSPACE / fname
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            found_data += 1
            logger.info(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            report.warn(f"Required data file missing: {fname}")

    # Check optional data files
    for fname in OPTIONAL_DATA_FILES:
        p = WORKSPACE / fname
        if p.exists():
            found_data += 1

    report.stats["data_files_found"] = found_data

    # Check core modules importable
    modules_ok = 0
    for mod in CORE_MODULES:
        try:
            __import__(mod)
            modules_ok += 1
        except ImportError as e:
            report.warn(f"Module '{mod}' not importable: {e}")

    report.stats["core_modules_ok"] = f"{modules_ok}/{len(CORE_MODULES)}"

    # Verify GOD_CODE invariant at multiple X checkpoints
    conservation_ok = True
    for x in CONSERVATION_CHECKPOINTS:
        g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
        w_x = 2 ** (x / 104)
        product = g_x * w_x
        if abs(product - GOD_CODE) > 1e-6:
            report.error(f"GOD_CODE conservation broken at X={x}: G(X)×W(X)={product:.10f}")
            conservation_ok = False
            ok = False
    if conservation_ok:
        logger.info(f"  ✓ GOD_CODE conservation verified at {len(CONSERVATION_CHECKPOINTS)} checkpoints")
    report.stats["conservation_checkpoints"] = len(CONSERVATION_CHECKPOINTS)

    # Workspace disk usage guard: warn if less than 50 MB free
    try:
        usage = shutil.disk_usage(str(WORKSPACE))
        free_mb = usage.free / (1024 * 1024)
        if free_mb < 50:
            report.warn(f"Low disk space: {free_mb:.0f} MB free")
    except OSError:
        pass

    report.add_phase("Preflight Checks", "OK" if ok else "FAIL", time.time() - t0,
                     f"{found_data} data files, {modules_ok} modules")
    return ok


def phase_detect_changes(report: KernelBuildReport, cache: BuildCache) -> Tuple[bool, List[Path]]:
    """Phase 2: Detect which data files changed since last build."""
    t0 = time.time()
    changed: List[Path] = []

    all_data = REQUIRED_DATA_FILES + OPTIONAL_DATA_FILES
    for fname in all_data:
        p = WORKSPACE / fname
        if p.exists() and cache.is_changed(p):
            changed.append(p)
            logger.info(f"  Δ Changed: {fname}")

    # Also check core .py source files
    for pyf in WORKSPACE.glob("l104_*.py"):
        if cache.is_changed(pyf):
            changed.append(pyf)

    needs_rebuild = len(changed) > 0
    status = "OK" if needs_rebuild else "SKIP"
    detail = f"{len(changed)} changed files" if needs_rebuild else "No changes detected"

    report.add_phase("Change Detection", status, time.time() - t0, detail)
    report.stats["changed_files"] = len(changed)
    return needs_rebuild, changed


def phase_build_kernel(report: KernelBuildReport, force: bool = False) -> bool:
    """Phase 3: Execute the full kernel build pipeline."""
    t0 = time.time()
    ok = True

    try:
        import build_full_kernel
        examples, params = build_full_kernel.main()

        report.stats["training_examples"] = params.get("num_examples", 0)
        report.stats["vocabulary_size"] = params.get("vocabulary_size", 0)
        report.stats["parameter_count"] = params.get("bag_of_words", 0)
        report.stats["parameter_count_human"] = f"{params.get('bag_of_words', 0) / 1e6:.1f}M"

    except Exception as e:
        report.error(f"Kernel build failed: {e}")
        ok = False

    report.add_phase("Kernel Build", "OK" if ok else "FAIL", time.time() - t0,
                     f"{report.stats.get('training_examples', '?')} examples")
    return ok


def phase_validate_artifacts(report: KernelBuildReport) -> bool:
    """Phase 4: Validate build outputs exist and are well-formed."""
    t0 = time.time()
    ok = True
    validated = 0

    # Check kernel_full_merged.jsonl
    merged = WORKSPACE / "kernel_full_merged.jsonl"
    if merged.exists():
        line_count = 0
        with open(merged) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "prompt" in obj and "completion" in obj:
                        line_count += 1
                except json.JSONDecodeError:
                    pass
        if line_count > 0:
            validated += 1
            logger.info(f"  ✓ kernel_full_merged.jsonl: {line_count:,} valid entries")
        else:
            report.warn("kernel_full_merged.jsonl has no valid entries")
    else:
        report.warn("kernel_full_merged.jsonl not produced")

    # Check KERNEL_MANIFEST.json
    manifest = WORKSPACE / "KERNEL_MANIFEST.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            required_keys = ["kernel_version", "total_examples", "vocabulary_size", "status"]
            missing = [k for k in required_keys if k not in data]
            if missing:
                report.warn(f"Manifest missing keys: {missing}")
            else:
                validated += 1
                logger.info(f"  ✓ KERNEL_MANIFEST.json: {data.get('status')}")
        except (json.JSONDecodeError, IOError) as e:
            report.warn(f"Manifest invalid: {e}")
    else:
        report.warn("KERNEL_MANIFEST.json not found")

    # Check fine_tune_exports directory
    ft_dir = WORKSPACE / "fine_tune_exports"
    if ft_dir.exists():
        export_count = len(list(ft_dir.glob("*.jsonl"))) + len(list(ft_dir.glob("*.json")))
        if export_count > 0:
            validated += 1
            logger.info(f"  ✓ fine_tune_exports/: {export_count} export files")
    else:
        report.warn("fine_tune_exports/ directory not found")

    report.stats["validated_artifacts"] = validated
    report.add_phase("Artifact Validation", "OK" if ok else "FAIL", time.time() - t0,
                     f"{validated} artifacts validated")
    return ok


def phase_update_cache(report: KernelBuildReport, cache: BuildCache):
    """Phase 5: Update build cache for incremental detection."""
    t0 = time.time()
    updated = 0

    all_files = REQUIRED_DATA_FILES + OPTIONAL_DATA_FILES
    for fname in all_files:
        p = WORKSPACE / fname
        if p.exists():
            cache.update(p)
            updated += 1

    for pyf in WORKSPACE.glob("l104_*.py"):
        cache.update(pyf)
        updated += 1

    cache.save()
    report.add_phase("Cache Update", "OK", time.time() - t0, f"{updated} files cached")


def phase_save_report(report: KernelBuildReport):
    """Phase 6: Save build report to disk."""
    t0 = time.time()
    report_path = BUILD_DIR / "last_build_report.json"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    report_data = report.to_dict()
    report_path.write_text(json.dumps(report_data, indent=2))

    # Also append to build history
    history_path = BUILD_DIR / "build_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": report_data["timestamp"],
            "success": report_data["success"],
            "elapsed_s": report_data["elapsed_s"],
            "examples": report.stats.get("training_examples", 0),
            "params": report.stats.get("parameter_count", 0),
        }) + "\n")

    # Rotate history if too large
    _rotate_history(history_path)

    report.add_phase("Save Report", "OK", time.time() - t0, str(report_path))


def _rotate_history(history_path: Path):
    """Trim build history to last MAX_HISTORY_ENTRIES entries."""
    if not history_path.exists():
        return
    try:
        lines = history_path.read_text().strip().splitlines()
        if len(lines) > MAX_HISTORY_ENTRIES:
            trimmed = lines[-MAX_HISTORY_ENTRIES:]
            history_path.write_text("\n".join(trimmed) + "\n")
            logger.info(f"Rotated build history: {len(lines)} → {len(trimmed)} entries")
    except (IOError, OSError) as e:
        logger.warning(f"History rotation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def build(force: bool = False, skip_build: bool = False, validate_only: bool = False) -> bool:
    """
    Run the L104 kernel build pipeline.

    Args:
        force: Force rebuild even if no changes detected
        skip_build: Skip the heavy full-kernel build, only check/validate
        validate_only: Only validate existing artifacts
    """
    report = KernelBuildReport()
    cache = BuildCache()
    lock = BuildLock()

    if not lock.acquire():
        report.error("Cannot acquire build lock — another build may be running")
        report.display()
        return False

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🧠 L104 KERNEL BUILDER                                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE: {GOD_CODE:.10f}   PHI: {PHI:.10f}                     ║
║   Mode: {"FORCE" if force else "VALIDATE-ONLY" if validate_only else "INCREMENTAL":<15s}                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    # Phase 1: Preflight
    logger.info("═══ PHASE 1: PREFLIGHT CHECKS ═══")
    if not phase_preflight(report, cache):
        report.display()
        return False

    if validate_only:
        logger.info("═══ VALIDATE-ONLY MODE ═══")
        phase_validate_artifacts(report)
        report.display()
        return len(report.errors) == 0

    # Phase 2: Change Detection
    logger.info("═══ PHASE 2: CHANGE DETECTION ═══")
    needs_rebuild, changed = phase_detect_changes(report, cache)

    # Phase 3: Build
    if needs_rebuild or force:
        if not skip_build:
            logger.info("═══ PHASE 3: KERNEL BUILD ═══")
            phase_build_kernel(report, force)
        else:
            report.add_phase("Kernel Build", "SKIP", 0.0, "Skipped by --skip-build")
    else:
        report.add_phase("Kernel Build", "SKIP", 0.0, "No changes detected (use --force)")

    # Phase 4: Validate
    logger.info("═══ PHASE 4: ARTIFACT VALIDATION ═══")
    phase_validate_artifacts(report)

    # Phase 5: Update Cache
    logger.info("═══ PHASE 5: CACHE UPDATE ═══")
    phase_update_cache(report, cache)

    # Phase 6: Save Report
    logger.info("═══ PHASE 6: SAVE REPORT ═══")
    phase_save_report(report)

    lock.release()
    report.display()
    return len(report.errors) == 0


def diff():
    """Compare last two build reports to show what changed."""
    report_path = BUILD_DIR / "last_build_report.json"
    history_path = BUILD_DIR / "build_history.jsonl"

    if not history_path.exists():
        print("No build history found. Run at least two builds first.")
        return

    lines = history_path.read_text().strip().splitlines()
    if len(lines) < 2:
        print("Need at least two builds to diff. Run another build first.")
        return

    prev = json.loads(lines[-2])
    curr = json.loads(lines[-1])

    print(f"\n═══ BUILD DIFF ═══")
    print(f"  Previous:  {prev.get('timestamp', '?')[:19]}")
    print(f"  Current:   {curr.get('timestamp', '?')[:19]}")
    print(f"  ─────────────────────────────────────")

    for key in ["success", "elapsed_s", "examples", "params"]:
        p_val = prev.get(key, "N/A")
        c_val = curr.get(key, "N/A")
        changed = " ←" if p_val != c_val else ""
        if isinstance(p_val, (int, float)) and isinstance(c_val, (int, float)):
            delta = c_val - p_val
            sign = "+" if delta > 0 else ""
            print(f"  {key:<15s}  {p_val!s:<12s} → {c_val!s:<12s} ({sign}{delta}){changed}")
        else:
            print(f"  {key:<15s}  {p_val!s:<12s} → {c_val!s:<12s}{changed}")
    print()


def clean():
    """Remove build cache and artifacts."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        logger.info(f"Cleaned build directory: {BUILD_DIR}")
    else:
        logger.info("Nothing to clean.")


def history():
    """Display build history."""
    history_path = BUILD_DIR / "build_history.jsonl"
    if not history_path.exists():
        print("No build history found. Run a build first.")
        return

    print(f"\n{'Timestamp':<28s} {'Status':<8s} {'Time':<8s} {'Examples':<10s} {'Params':<12s}")
    print("─" * 70)
    with open(history_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                status = "✓ OK" if entry["success"] else "✗ FAIL"
                ts = entry["timestamp"][:19]
                elapsed = f"{entry['elapsed_s']:.1f}s"
                examples = f"{entry.get('examples', 0):,}"
                params = f"{entry.get('params', 0):,}"
                print(f"  {ts:<26s} {status:<8s} {elapsed:<8s} {examples:<10s} {params:<12s}")
            except (json.JSONDecodeError, KeyError):
                pass
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="L104 Kernel Builder - Build, validate, and manage kernel artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python build_kernel.py                  # Incremental build (skips if unchanged)
  python build_kernel.py --force          # Force full rebuild
  python build_kernel.py --validate       # Only validate existing artifacts
  python build_kernel.py --skip-build     # Check & validate without heavy build
  python build_kernel.py clean            # Remove build cache
  python build_kernel.py history          # Show build history

GOD_CODE = {GOD_CODE}
        """,
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("clean", help="Remove build cache and temporary artifacts")
    subparsers.add_parser("history", help="Display build history")
    subparsers.add_parser("diff", help="Compare last two builds")

    parser.add_argument("--force", "-f", action="store_true", help="Force full rebuild")
    parser.add_argument("--validate", "-v", action="store_true", help="Only validate artifacts")
    parser.add_argument("--skip-build", "-s", action="store_true", help="Skip heavy kernel build")
    parser.add_argument("--checksum", action="store_true",
                        help="Print SHA-256 checksums of all kernel artifacts")

    args = parser.parse_args()

    if args.command == "clean":
        clean()
    elif args.command == "history":
        history()
    elif args.command == "diff":
        diff()
    elif getattr(args, 'checksum', False):
        print("\n═══ KERNEL ARTIFACT CHECKSUMS ═══")
        artifacts = [
            WORKSPACE / "kernel_full_merged.jsonl",
            WORKSPACE / "KERNEL_MANIFEST.json",
            WORKSPACE / "kernel_parameters.json",
        ]
        for art in artifacts:
            if art.exists():
                h = hashlib.sha256(art.read_bytes()).hexdigest()
                print(f"  {art.name:<35s} {h[:16]}...{h[-8:]}")
            else:
                print(f"  {art.name:<35s} (missing)")
        print()
    else:
        success = build(
            force=args.force,
            skip_build=args.skip_build,
            validate_only=args.validate,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
