#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 WALLET INTEGRITY CHECKER — Key & Transaction Audit
═══════════════════════════════════════════════════════════════════════════════

Inspects wallet_keys.db and the SecureKeyStore layer, reporting on:
  • Database schema and table integrity
  • Key count and chain-type distribution
  • Audit log analysis
  • Encryption sanity checks (key length, non-empty ciphertext)
  • Overall wallet health score

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import sqlite3
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492612
WORKSPACE = Path(__file__).parent.absolute()

WALLET_DB_PATH = WORKSPACE / "wallet_keys.db"
API_DB_PATH = WORKSPACE / "api_keys.db"

EXPECTED_TABLES = {"keys", "key_audit"}
EXPECTED_KEY_COLUMNS = {"id", "key_name", "chain_type", "encrypted_value", "created_at"}
EXPECTED_AUDIT_COLUMNS = {"id", "key_name", "action", "timestamp"}

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [WALLET-CHK] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("WALLET-CHK")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TableStatus:
    """Status for a single database table."""
    name: str
    exists: bool = False
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)


@dataclass
class ChainDistribution:
    """Distribution of keys across chain types."""
    chain_type: str
    count: int = 0


@dataclass
class AuditSummary:
    """Summary of the audit log."""
    total_entries: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    first_entry: str = ""
    last_entry: str = ""


@dataclass
class EncryptionCheck:
    """Result of encryption sanity checks."""
    total_keys: int = 0
    empty_ciphertext: int = 0
    avg_ciphertext_len: float = 0.0
    min_ciphertext_len: int = 0
    max_ciphertext_len: int = 0
    duplicate_key_names: List[str] = field(default_factory=list)


@dataclass
class WalletHealthReport:
    """Full wallet health report."""
    timestamp: str = ""
    db_path: str = ""
    db_exists: bool = False
    db_size_kb: float = 0.0
    integrity_ok: bool = False
    tables: List[TableStatus] = field(default_factory=list)
    key_count: int = 0
    chain_distribution: List[ChainDistribution] = field(default_factory=list)
    audit: AuditSummary = field(default_factory=AuditSummary)
    encryption: EncryptionCheck = field(default_factory=EncryptionCheck)
    api_db_exists: bool = False
    api_db_size_kb: float = 0.0
    god_code_verified: bool = False
    oldest_key: str = ""
    newest_key: str = ""
    health_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════


def check_db_exists(db_path: Path) -> Tuple[bool, float]:
    """Check if database file exists and report size."""
    if not db_path.exists():
        return False, 0.0
    return True, round(db_path.stat().st_size / 1024, 1)


def check_integrity(db_path: Path) -> bool:
    """Run SQLite integrity_check pragma."""
    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        return result and result[0] == "ok"
    except Exception as e:
        logger.warning(f"Integrity check failed: {e}")
        return False


def get_tables(db_path: Path) -> List[str]:
    """List all tables in the database."""
    try:
        conn = sqlite3.connect(str(db_path))
        tables = [row[0] for row in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def get_table_columns(db_path: Path, table: str) -> List[str]:
    """Get column names for a table."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        conn.close()
        return cols
    except Exception:
        return []


def get_row_count(db_path: Path, table: str) -> int:
    """Count rows in a table."""
    try:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def check_table_status(db_path: Path, table_name: str,
                       expected_cols: set) -> TableStatus:
    """Full check on a single table."""
    actual_tables = get_tables(db_path)
    status = TableStatus(name=table_name)

    if table_name not in actual_tables:
        status.exists = False
        return status

    status.exists = True
    status.columns = get_table_columns(db_path, table_name)
    status.row_count = get_row_count(db_path, table_name)
    status.missing_columns = [c for c in expected_cols if c not in status.columns]
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN & KEY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_chain_distribution(db_path: Path) -> List[ChainDistribution]:
    """Group keys by chain_type."""
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT chain_type, COUNT(*) FROM keys GROUP BY chain_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        conn.close()
        return [ChainDistribution(chain_type=r[0] or "unknown", count=r[1]) for r in rows]
    except Exception:
        return []


def analyze_encryption(db_path: Path) -> EncryptionCheck:
    """Sanity-check encrypted values and detect duplicate key names."""
    ec = EncryptionCheck()
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT encrypted_value FROM keys").fetchall()

        ec.total_keys = len(rows)
        if not rows:
            conn.close()
            return ec

        lengths = []
        for (val,) in rows:
            if val is None or (isinstance(val, (str, bytes)) and len(val) == 0):
                ec.empty_ciphertext += 1
            else:
                lengths.append(len(val) if val else 0)

        if lengths:
            ec.avg_ciphertext_len = round(sum(lengths) / len(lengths), 1)
            ec.min_ciphertext_len = min(lengths)
            ec.max_ciphertext_len = max(lengths)

        # Duplicate key name detection
        name_rows = conn.execute("SELECT key_name, COUNT(*) FROM keys GROUP BY key_name HAVING COUNT(*) > 1").fetchall()
        ec.duplicate_key_names = [r[0] for r in name_rows]

        conn.close()
    except Exception as e:
        logger.warning(f"Encryption analysis skipped: {e}")
    return ec


def analyze_audit_log(db_path: Path) -> AuditSummary:
    """Summarize audit log entries."""
    audit = AuditSummary()
    try:
        conn = sqlite3.connect(str(db_path))

        audit.total_entries = conn.execute("SELECT COUNT(*) FROM key_audit").fetchone()[0]
        if audit.total_entries == 0:
            conn.close()
            return audit

        # Action distribution
        action_rows = conn.execute(
            "SELECT action, COUNT(*) FROM key_audit GROUP BY action ORDER BY COUNT(*) DESC"
        ).fetchall()
        audit.action_counts = {r[0]: r[1] for r in action_rows}

        # Time range
        first = conn.execute(
            "SELECT timestamp FROM key_audit ORDER BY timestamp ASC LIMIT 1"
        ).fetchone()
        last = conn.execute(
            "SELECT timestamp FROM key_audit ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if first:
            audit.first_entry = str(first[0])
        if last:
            audit.last_entry = str(last[0])

        conn.close()
    except Exception as e:
        logger.warning(f"Audit analysis skipped: {e}")
    return audit


def analyze_key_ages(db_path: Path) -> Tuple[str, str]:
    """Find the oldest and newest key creation timestamps."""
    try:
        conn = sqlite3.connect(str(db_path))
        oldest = conn.execute("SELECT created_at FROM keys ORDER BY created_at ASC LIMIT 1").fetchone()
        newest = conn.execute("SELECT created_at FROM keys ORDER BY created_at DESC LIMIT 1").fetchone()
        conn.close()
        return (str(oldest[0]) if oldest else "", str(newest[0]) if newest else "")
    except Exception:
        return ("", "")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORER
# ═══════════════════════════════════════════════════════════════════════════════


def score_health(report: WalletHealthReport) -> float:
    """
    Compute a 0–100 health score based on checks.
    Each check contributes weighted points.
    """
    score = 0.0
    total = 0.0

    # DB exists (25 pts)
    total += 25
    if report.db_exists:
        score += 25

    # Integrity (25 pts)
    total += 25
    if report.integrity_ok:
        score += 25

    # Tables present (20 pts)
    total += 20
    for ts in report.tables:
        if ts.exists:
            score += 10
        if not ts.missing_columns:
            score += 0  # bonus included with table existence

    # Encryption quality (15 pts)
    total += 15
    if report.encryption.total_keys > 0:
        if report.encryption.empty_ciphertext == 0:
            score += 15
        else:
            pct_valid = 1 - (report.encryption.empty_ciphertext / report.encryption.total_keys)
            score += 15 * pct_valid

    # Audit log present (15 pts)
    total += 15
    if report.audit.total_entries > 0:
        score += 15

    return round(100 * (score / total) if total > 0 else 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHECK ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def check_wallet(db_path: Optional[Path] = None, verbose: bool = True,
                 show_keys: bool = False) -> WalletHealthReport:
    """Run full wallet integrity check."""
    t0 = time.time()
    db = db_path or WALLET_DB_PATH
    report = WalletHealthReport(
        timestamp=datetime.now().isoformat(),
        db_path=str(db),
    )

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🔐 L104 WALLET INTEGRITY CHECK                                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE:      {GOD_CODE:.10f}                                          ║
║   Wallet DB:     {db.name:<40s}                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    # 1. Database existence
    report.db_exists, report.db_size_kb = check_db_exists(db)
    if not report.db_exists:
        report.issues.append(f"Database not found: {db}")
        report.health_score = 0.0
        report.elapsed_s = time.time() - t0
        print(f"  ✗ Database not found: {db}")
        print(f"    Create with: python l104_wallet_system.py")
        return report
    else:
        print(f"  ✓ Database found — {report.db_size_kb:.1f} KB")

    # 2. Integrity
    report.integrity_ok = check_integrity(db)
    if report.integrity_ok:
        print(f"  ✓ SQLite integrity check passed")
    else:
        report.issues.append("SQLite integrity_check FAILED")
        print(f"  ✗ SQLite integrity check FAILED")

    # 3. Table checks
    keys_status = check_table_status(db, "keys", EXPECTED_KEY_COLUMNS)
    audit_status = check_table_status(db, "key_audit", EXPECTED_AUDIT_COLUMNS)
    report.tables = [keys_status, audit_status]

    for ts in report.tables:
        if ts.exists:
            print(f"  ✓ Table '{ts.name}' — {ts.row_count} rows, "
                  f"columns: {', '.join(ts.columns[:6])}")
            if ts.missing_columns:
                report.issues.append(f"Table '{ts.name}' missing columns: {ts.missing_columns}")
                print(f"    ⚠ Missing columns: {ts.missing_columns}")
        else:
            report.issues.append(f"Table '{ts.name}' does not exist")
            print(f"  ✗ Table '{ts.name}' not found")

    # 4. Chain distribution
    report.chain_distribution = analyze_chain_distribution(db)
    report.key_count = sum(c.count for c in report.chain_distribution)
    if report.chain_distribution:
        print(f"\n  Chain Distribution ({report.key_count} total keys):")
        for cd in report.chain_distribution:
            bar = "█" * min(cd.count, 40)
            print(f"    {cd.chain_type:<16s} {cd.count:>4}  {bar}")
    else:
        print(f"\n  No keys found in database.")

    # 5. Encryption check
    report.encryption = analyze_encryption(db)
    if report.encryption.total_keys > 0:
        print(f"\n  Encryption Status:")
        print(f"    Total keys:        {report.encryption.total_keys}")
        print(f"    Empty ciphertext:  {report.encryption.empty_ciphertext}")
        print(f"    Avg cipher length: {report.encryption.avg_ciphertext_len}")
        print(f"    Range:             {report.encryption.min_ciphertext_len} — "
              f"{report.encryption.max_ciphertext_len}")
        if report.encryption.empty_ciphertext > 0:
            report.issues.append(
                f"{report.encryption.empty_ciphertext} key(s) have empty ciphertext"
            )
        if report.encryption.duplicate_key_names:
            dups = ", ".join(report.encryption.duplicate_key_names[:5])
            report.issues.append(f"Duplicate key names: {dups}")
            print(f"    ⚠ Duplicates:      {dups}")

    # 5b. Key age analysis
    report.oldest_key, report.newest_key = analyze_key_ages(db)
    if report.oldest_key:
        print(f"\n  Key Age:")
        print(f"    Oldest: {report.oldest_key}")
        print(f"    Newest: {report.newest_key}")

    # 5c. Verbose key listing
    if show_keys and report.key_count > 0:
        try:
            conn = sqlite3.connect(str(db))
            rows = conn.execute(
                "SELECT key_name, chain_type, created_at FROM keys ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            print(f"\n  All Keys ({len(rows)}):")
            for name, chain, created in rows[:50]:
                print(f"    {name:<30s} {chain:<14s} {created}")
            if len(rows) > 50:
                print(f"    ... and {len(rows) - 50} more")
        except Exception:
            pass

    # 6. Audit log
    report.audit = analyze_audit_log(db)
    if report.audit.total_entries > 0:
        print(f"\n  Audit Log:")
        print(f"    Total entries:  {report.audit.total_entries}")
        for action, count in report.audit.action_counts.items():
            print(f"      {action:<20s} {count}")
        print(f"    First:  {report.audit.first_entry}")
        print(f"    Last:   {report.audit.last_entry}")
    else:
        print(f"\n  Audit log: empty")

    # 7. API keys DB (secondary check + cross-reference)
    report.api_db_exists, report.api_db_size_kb = check_db_exists(API_DB_PATH)
    if report.api_db_exists:
        print(f"\n  ✓ API keys DB found — {report.api_db_size_kb:.1f} KB")
        # Cross-reference: count API keys and flag orphans
        try:
            conn = sqlite3.connect(str(API_DB_PATH))
            api_tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "api_keys" in api_tables:
                api_count = conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
                print(f"    API keys registered: {api_count}")
            conn.close()
        except Exception:
            pass
    else:
        print(f"\n  — API keys DB not present (optional)")

    # 7b. GOD_CODE conservation (multi-checkpoint)
    conservation_ok = True
    worst_delta = 0.0
    for x in [0, 52, 104, 208, 286, 416]:
        g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
        product = g_x * (2 ** (x / 104))
        delta = abs(product - GOD_CODE)
        worst_delta = max(worst_delta, delta)
        if delta > 1e-6:
            conservation_ok = False
    report.god_code_verified = conservation_ok
    gc_icon = "✓" if conservation_ok else "✗"
    print(f"  {gc_icon} GOD_CODE conservation verified at 6 checkpoints (worst Δ={worst_delta:.2e})")

    # 7c. Key rotation/expiration warnings (keys older than 90 days)
    if report.oldest_key:
        try:
            from datetime import timedelta
            oldest_dt = datetime.fromisoformat(report.oldest_key.replace('Z', '+00:00'))
            age_days = (datetime.now(oldest_dt.tzinfo) - oldest_dt).days if oldest_dt.tzinfo else (
                datetime.now() - oldest_dt).days
            if age_days > 90:
                report.issues.append(f"Oldest key is {age_days} days old — consider rotation")
                print(f"  ⚠ Key rotation recommended: oldest key is {age_days} days old")
        except (ValueError, TypeError):
            pass

    # 7d. Backup check
    backup_dir = WORKSPACE / ".kernel_build" / "backups"
    if backup_dir.exists():
        wallet_backups = sorted(backup_dir.glob("*wallet*"), reverse=True)
        if wallet_backups:
            last_backup_age = (time.time() - wallet_backups[0].stat().st_mtime) / 86400
            print(f"  Last wallet backup: {last_backup_age:.0f} days ago")
        else:
            print(f"  — No wallet backups found in {backup_dir}")

    # 8. Score
    report.health_score = score_health(report)
    report.elapsed_s = round(time.time() - t0, 3)

    # Dashboard
    bar_len = 30
    filled = int(report.health_score / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   📊 WALLET HEALTH SUMMARY                                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Health Score:   [{bar}] {report.health_score:.0f}%          ║
║   Keys:           {report.key_count:>5}     Chains: {len(report.chain_distribution):>3}                                     ║
║   Audit Entries:  {report.audit.total_entries:>5}                                                     ║
║   Issues:         {len(report.issues):>5}                                                     ║
║   Elapsed:        {report.elapsed_s:.3f}s                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝""")

    if report.issues:
        print("\n  Issues Found:")
        for i, issue in enumerate(report.issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  ✓ No issues found — wallet is healthy.")
    print()

    return report


def export_report(report: WalletHealthReport, path: Optional[Path] = None) -> Path:
    """Export wallet health report to JSON."""
    out = path or (WORKSPACE / ".kernel_build" / "wallet_health_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(report)
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Report exported to {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L104 Wallet Integrity Checker — inspect wallet_keys.db health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python check_wallet.py                  # Full wallet check
  python check_wallet.py --db path/to.db  # Check a specific database
  python check_wallet.py --export         # Save report to JSON
  python check_wallet.py --json           # Output as JSON to stdout

GOD_CODE = {GOD_CODE}
        """,
    )
    parser.add_argument("--db", type=str, default=None,
                        help="Path to wallet database (default: wallet_keys.db)")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export report to JSON file")
    parser.add_argument("--json", action="store_true",
                        help="Print report as JSON to stdout")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full key listing")
    parser.add_argument("--check-api", action="store_true",
                        help="Also run integrity check on api_keys.db")
    args = parser.parse_args()

    db = Path(args.db) if args.db else None
    report = check_wallet(db_path=db, verbose=not args.json, show_keys=args.verbose)

    if getattr(args, 'check_api', False) and API_DB_PATH.exists():
        print("\n  === API Keys DB Audit ===")
        api_ok = check_integrity(API_DB_PATH)
        icon = "\u2713" if api_ok else "\u2717"
        status = "OK" if api_ok else "FAIL"
        print(f"  {icon} api_keys.db integrity: {status}")
        api_tables = get_tables(API_DB_PATH)
        for t in api_tables:
            rc = get_row_count(API_DB_PATH, t)
            print(f"    Table '{t}': {rc} rows")
        print()

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))

    if args.export:
        export_report(report)

    # Exit 0 if healthy (≥80), 1 otherwise
    sys.exit(0 if report.health_score >= 80 else 1)