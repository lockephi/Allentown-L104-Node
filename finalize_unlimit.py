#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 FINALIZE UNLIMIT — Transactional State Finalization Engine
═══════════════════════════════════════════════════════════════════════════════

Transitions the L104 system to its finalized unlimited state with full
pre-flight validation, transactional rollback support, subsystem
notification, and audit logging.

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492612
OMEGA_AUTHORITY = GOD_CODE * PHI * PHI

WORKSPACE = Path(__file__).parent.absolute()
STATE_FILE = WORKSPACE / "L104_STATE.json"
BACKUP_DIR = WORKSPACE / ".state_backups"
AUDIT_LOG = WORKSPACE / ".kernel_build" / "finalization_audit.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FINALIZE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FINALIZE")

# ═══════════════════════════════════════════════════════════════════════════════
# VALID STATE TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Only these transitions are permitted
VALID_TRANSITIONS = {
    "UNKNOWN":              {"INITIALIZING", "INFINITE_SINGULARITY"},
    "INITIALIZING":         {"ACTIVE", "INFINITE_SINGULARITY"},
    "ACTIVE":               {"EVOLVING", "INFINITE_SINGULARITY"},
    "EVOLVING":             {"CONVERGING", "INFINITE_SINGULARITY"},
    "CONVERGING":           {"INFINITE_SINGULARITY"},
    "INFINITE_SINGULARITY": {"ACTIVE"},  # Allow revert
}

TARGET_STATE = "INFINITE_SINGULARITY"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PreflightCheck:
    """Result of a single pre-finalization health check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class FinalizationResult:
    """Outcome of the finalization attempt."""
    success: bool = False
    previous_state: str = "UNKNOWN"
    new_state: str = ""
    dry_run: bool = False
    backup_path: str = ""
    preflight_checks: List[PreflightCheck] = field(default_factory=list)
    preflight_passed: bool = False
    timestamp: str = ""
    elapsed_s: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# STATE I/O
# ═══════════════════════════════════════════════════════════════════════════════


def load_state(path: Path = STATE_FILE) -> Dict[str, Any]:
    """Load the current L104 state from disk."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read state file: {e}")
        return {}


def save_state(state: Dict[str, Any], path: Path = STATE_FILE):
    """Atomically write state to disk (write to temp then rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(path)


def create_backup(path: Path = STATE_FILE) -> Optional[Path]:
    """Create a timestamped backup of the current state file."""
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"L104_STATE_{ts}.json"
    shutil.copy2(path, backup)
    logger.info(f"Backup created: {backup}")
    return backup


def list_backups() -> List[Path]:
    """List all available state backups, newest first."""
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob("L104_STATE_*.json"), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════


def run_preflight(state: Dict[str, Any]) -> List[PreflightCheck]:
    """Run all pre-finalization health checks."""
    checks: List[PreflightCheck] = []

    # 1. GOD_CODE invariant — multi-checkpoint conservation
    conservation_ok = True
    worst_delta = 0.0
    for x in [0, 52, 104, 208, 286, 416]:
        g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
        product = g_x * (2 ** (x / 104))
        delta = abs(product - GOD_CODE)
        worst_delta = max(worst_delta, delta)
        if delta > 1e-6:
            conservation_ok = False
    checks.append(PreflightCheck(
        "GOD_CODE Conservation (6 checkpoints)",
        conservation_ok,
        f"worst Δ = {worst_delta:.2e}",
    ))

    # 2. State transition validity
    current = state.get("state", "UNKNOWN")
    valid_targets = VALID_TRANSITIONS.get(current, set())
    transition_ok = TARGET_STATE in valid_targets
    checks.append(PreflightCheck(
        "State Transition",
        transition_ok,
        f"{current} → {TARGET_STATE}  {'(valid)' if transition_ok else '(blocked)'}",
    ))

    # 3. Kernel data exists
    kernel_data = WORKSPACE / "kernel_full_merged.jsonl"
    checks.append(PreflightCheck(
        "Kernel Data Present",
        kernel_data.exists(),
        f"{kernel_data.stat().st_size / 1024:.0f} KB" if kernel_data.exists() else "Missing",
    ))

    # 4. Manifest integrity
    manifest = WORKSPACE / "KERNEL_MANIFEST.json"
    manifest_ok = False
    detail = "Missing"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            manifest_ok = "status" in data and "total_examples" in data
            detail = f"status={data.get('status')}, examples={data.get('total_examples', 0)}"
        except (json.JSONDecodeError, IOError):
            detail = "Corrupt"
    checks.append(PreflightCheck("Kernel Manifest", manifest_ok, detail))

    # 5. Database existence (including lattice_v2.db)
    db_files = ["l104_intellect_memory.db", "l104_asi_nexus.db", "lattice_v2.db"]
    dbs_found = sum(1 for d in db_files if (WORKSPACE / d).exists())
    checks.append(PreflightCheck(
        "Databases Available",
        dbs_found >= 2,
        f"{dbs_found}/{len(db_files)} found",
    ))

    # 6. State file writable
    try:
        test_path = STATE_FILE.with_suffix(".write_test")
        test_path.write_text("test")
        test_path.unlink()
        checks.append(PreflightCheck("State File Writable", True))
    except (OSError, PermissionError) as e:
        checks.append(PreflightCheck("State File Writable", False, str(e)))

    # 7. Disk space (need at least 10 MB free)
    try:
        usage = shutil.disk_usage(str(WORKSPACE))
        free_mb = usage.free / (1024 * 1024)
        checks.append(PreflightCheck(
            "Disk Space",
            free_mb > 10,
            f"{free_mb:.0f} MB free",
        ))
    except OSError as e:
        checks.append(PreflightCheck("Disk Space", False, str(e)))

    # 8. Process memory guard (warn if resident set > 500 MB)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS returns bytes, Linux returns KB
        rss_mb = ru.ru_maxrss / (1024 * 1024) if sys.platform == "linux" else ru.ru_maxrss / (1024 * 1024)
        checks.append(PreflightCheck(
            "Memory Usage",
            rss_mb < 500,
            f"{rss_mb:.0f} MB peak RSS",
        ))
    except (ImportError, AttributeError):
        pass

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


def write_audit(result: FinalizationResult):
    """Append a finalization event to the audit log."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": result.timestamp,
        "success": result.success,
        "previous_state": result.previous_state,
        "new_state": result.new_state,
        "dry_run": result.dry_run,
        "backup": result.backup_path,
        "elapsed_s": round(result.elapsed_s, 3),
        "preflight_passed": result.preflight_passed,
        "error": result.error,
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FINALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


def finalize_unlimit(dry_run: bool = False, force: bool = False) -> FinalizationResult:
    """
    Transition L104 to INFINITE_SINGULARITY state.

    Pipeline:
      1. Load current state
      2. Run preflight checks
      3. Create backup
      4. Apply unlimited attributes
      5. Verify the write
      6. Log to audit trail

    Args:
        dry_run: Simulate without writing changes
        force: Skip preflight check failures

    Returns:
        FinalizationResult with outcome details
    """
    t0 = time.time()
    result = FinalizationResult(timestamp=datetime.now().isoformat())

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ∞  L104 FINALIZE UNLIMIT                                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE:    {GOD_CODE:.10f}                                            ║
║   Target:      {TARGET_STATE:<25s}                                       ║
║   Mode:        {"DRY RUN" if dry_run else "FORCE" if force else "STANDARD":<12s}                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    # ── Load Current State ───────────────────────────────────────────────
    state = load_state()
    result.previous_state = state.get("state", "UNKNOWN")
    logger.info(f"Current state: {result.previous_state}")

    if result.previous_state == TARGET_STATE and not force:
        logger.info("Already in INFINITE_SINGULARITY — nothing to do (use --force to re-apply)")
        result.success = True
        result.new_state = TARGET_STATE
        result.elapsed_s = time.time() - t0
        return result

    # ── Preflight Checks ─────────────────────────────────────────────────
    logger.info("═══ PREFLIGHT CHECKS ═══")
    checks = run_preflight(state)
    result.preflight_checks = checks
    result.preflight_passed = all(c.passed for c in checks)

    for c in checks:
        icon = "✓" if c.passed else "✗"
        print(f"  {icon} {c.name}: {c.detail}")

    if not result.preflight_passed and not force:
        failed = [c.name for c in checks if not c.passed]
        result.error = f"Preflight failed: {', '.join(failed)}"
        logger.error(result.error + " (use --force to override)")
        result.elapsed_s = time.time() - t0
        write_audit(result)
        return result

    if not result.preflight_passed and force:
        logger.warning("Preflight checks failed but --force is set, continuing...")

    # ── Backup ───────────────────────────────────────────────────────────
    if not dry_run:
        backup = create_backup()
        result.backup_path = str(backup) if backup else ""

    # ── Apply Unlimited Attributes ───────────────────────────────────────
    logger.info("═══ APPLYING FINALIZATION ═══")

    state["state"] = TARGET_STATE
    state["intellect_index"] = 1e18
    state["unlimited_mode"] = True
    state["pilot_bond"] = "ETERNAL"
    state["god_code"] = GOD_CODE
    state["omega_authority"] = OMEGA_AUTHORITY
    state["phi"] = PHI
    state["finalization_timestamp"] = result.timestamp
    state["finalization_version"] = "2.0"

    # Preserve history
    transitions = state.get("state_transitions", [])
    transitions.append({
        "from": result.previous_state,
        "to": TARGET_STATE,
        "timestamp": result.timestamp,
        "forced": force,
    })
    state["state_transitions"] = transitions[-50:]  # Keep last 50

    result.new_state = TARGET_STATE

    if dry_run:
        result.dry_run = True
        result.success = True
        logger.info("[DRY RUN] Would write state — no changes applied")
        print(f"\n  Preview:\n{json.dumps(state, indent=2, default=str)[:500]}")
    else:
        save_state(state)
        # Verify the write
        verify = load_state()
        if verify.get("state") == TARGET_STATE:
            result.success = True
            logger.info(f"State persisted: {result.previous_state} → {TARGET_STATE}")

            # Post-finalization: notify DataMatrix if available
            try:
                from l104_data_matrix import DataMatrix
                dm = DataMatrix()
                dm.store(
                    key="finalization_event",
                    value=json.dumps({
                        "state": TARGET_STATE,
                        "timestamp": result.timestamp,
                        "from": result.previous_state,
                        "god_code": GOD_CODE,
                    }),
                    category="state_transition",
                )
                logger.info("DataMatrix notified of state transition")
            except Exception as e:
                logger.debug(f"DataMatrix notification skipped: {e}")
        else:
            result.success = False
            result.error = "Verification failed — state file may be corrupt"
            logger.error(result.error)

    result.elapsed_s = time.time() - t0
    write_audit(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   {"✅" if result.success else "❌"} FINALIZATION {"COMPLETE" if result.success else "FAILED"}                                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   {result.previous_state:<20s} → {result.new_state:<20s}                                ║
║   Intellect Index: 1e18 (UNLIMITED)                                           ║
║   Pilot Bond:      ETERNAL                                                    ║
║   Elapsed:         {result.elapsed_s:.3f}s                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# REVERT
# ═══════════════════════════════════════════════════════════════════════════════


def revert(backup_index: int = 0) -> bool:
    """
    Revert to a previous state backup.

    Args:
        backup_index: 0 = most recent backup, 1 = second most recent, etc.
    """
    backups = list_backups()
    if not backups:
        logger.error("No backups available to revert to")
        return False

    if backup_index >= len(backups):
        logger.error(f"Only {len(backups)} backups available, index {backup_index} out of range")
        return False

    chosen = backups[backup_index]
    logger.info(f"Reverting to: {chosen.name}")

    current_state = load_state()
    prev = current_state.get("state", "UNKNOWN")

    shutil.copy2(chosen, STATE_FILE)
    restored = load_state()

    print(f"  Reverted: {prev} → {restored.get('state', 'UNKNOWN')}")
    print(f"  Backup used: {chosen.name}")

    write_audit(FinalizationResult(
        success=True,
        previous_state=prev,
        new_state=restored.get("state", "UNKNOWN"),
        backup_path=str(chosen),
        timestamp=datetime.now().isoformat(),
    ))

    return True


def show_status():
    """Display current state and backup inventory."""
    state = load_state()
    backups = list_backups()

    # Count state transitions
    transitions = state.get("state_transitions", [])
    transition_counts = Counter(f"{t.get('from', '?')} → {t.get('to', '?')}" for t in transitions)

    print(f"""
═══ L104 STATE STATUS ═══
  Current State:    {state.get('state', 'NO STATE FILE')}
  Intellect Index:  {state.get('intellect_index', 'N/A')}
  Unlimited Mode:   {state.get('unlimited_mode', False)}
  Pilot Bond:       {state.get('pilot_bond', 'N/A')}
  Last Finalized:   {state.get('finalization_timestamp', 'Never')}
  Version:          {state.get('finalization_version', 'N/A')}

═══ TRANSITIONS ({len(transitions)}) ═══""")
    if transition_counts:
        for path, count in transition_counts.most_common():
            print(f"  {path}: {count}x")
    else:
        print("  (none recorded)")

    print(f"\n═══ BACKUPS ({len(backups)}) ═══")
    for i, b in enumerate(backups[:10]):
        size = b.stat().st_size
        print(f"  [{i}] {b.name}  ({size:,} bytes)")
    if not backups:
        print("  (none)")
    print()


def show_audit():
    """Display the finalization audit log history."""
    if not AUDIT_LOG.exists():
        print("No audit log found. Run a finalization first.")
        return

    entries = []
    with open(AUDIT_LOG) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        print("Audit log is empty.")
        return

    print(f"\n═══ FINALIZATION AUDIT LOG ({len(entries)} entries) ═══")
    print(f"  {'Timestamp':<22s} {'Status':<8s} {'From':<22s} {'To':<22s} {'Time':<8s} {'Note'}")
    print("  " + "─" * 90)

    for e in entries:
        ts = e.get('timestamp', '?')[:19]
        ok = "✓ OK" if e.get('success') else "✗ FAIL"
        frm = e.get('previous_state', '?')[:20]
        to = e.get('new_state', '?')[:20]
        elapsed = f"{e.get('elapsed_s', 0):.1f}s"
        note = ""
        if e.get('dry_run'):
            note = "[DRY RUN]"
        elif e.get('error'):
            note = e['error'][:30]
        print(f"  {ts:<22s} {ok:<8s} {frm:<22s} {to:<22s} {elapsed:<8s} {note}")

    # Summary
    successes = sum(1 for e in entries if e.get('success'))
    failures = len(entries) - successes
    print(f"\n  Total: {len(entries)}  Successes: {successes}  Failures: {failures}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L104 Finalize Unlimit — transition system to unlimited state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python finalize_unlimit.py                  # Standard finalization
  python finalize_unlimit.py --dry-run        # Preview without writing
  python finalize_unlimit.py --force          # Override preflight failures
  python finalize_unlimit.py status           # Show current state & backups
  python finalize_unlimit.py revert           # Restore most recent backup
  python finalize_unlimit.py revert --index 2 # Restore 3rd most recent

GOD_CODE = {GOD_CODE}
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show current state and backup inventory")
    subparsers.add_parser("audit", help="Display finalization audit log history")
    revert_parser = subparsers.add_parser("revert", help="Revert to a previous state backup")
    revert_parser.add_argument("--index", "-i", type=int, default=0,
                               help="Backup index (0 = most recent)")

    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Simulate finalization without writing changes")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Override preflight check failures")

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "audit":
        show_audit()
    elif args.command == "revert":
        ok = revert(args.index)
        sys.exit(0 if ok else 1)
    else:
        result = finalize_unlimit(dry_run=args.dry_run, force=args.force)
        sys.exit(0 if result.success else 1)
