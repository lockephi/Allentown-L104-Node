#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 FINAL REPORT GENERATOR — Dynamic System Intelligence Report
═══════════════════════════════════════════════════════════════════════════════

Collects real metrics from all L104 subsystems — databases, kernel,
knowledge base, memory, APIs — and produces a comprehensive status report
with computed (not hardcoded) statistics.

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import sys
import os
import json
import time
import sqlite3
import platform
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492612
OMEGA_AUTHORITY = GOD_CODE * PHI * PHI
WORKSPACE = Path(__file__).parent.absolute()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REPORT] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("REPORT")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SubsystemMetrics:
    """Metrics collected from a single subsystem."""
    name: str
    status: str = "UNKNOWN"
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class FullReport:
    """Complete L104 system intelligence report."""
    timestamp: str = ""
    pilot: str = "LONDEL"
    god_code: float = GOD_CODE
    god_code_verified: bool = False
    omega_authority: float = OMEGA_AUTHORITY
    environment: Dict[str, str] = field(default_factory=dict)
    core_status: Dict[str, Any] = field(default_factory=dict)
    database_metrics: List[SubsystemMetrics] = field(default_factory=list)
    kernel_metrics: Dict[str, Any] = field(default_factory=dict)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    file_inventory: Dict[str, Any] = field(default_factory=dict)
    subsystem_modules: List[SubsystemMetrics] = field(default_factory=list)
    elapsed_s: float = 0.0
    previous_comparison: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTORS
# ═══════════════════════════════════════════════════════════════════════════════


def collect_environment() -> Dict[str, str]:
    """Collect runtime environment details."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "node": platform.node(),
        "workspace": str(WORKSPACE),
        "pid": str(os.getpid()),
    }


def collect_core_status() -> Dict[str, Any]:
    """Attempt to load AGI/ASI cores and read real metrics."""
    result: Dict[str, Any] = {}

    # AGI Core
    try:
        from l104_agi_core import AGICore
        agi = AGICore()
        result["agi"] = {
            "core_type": getattr(agi, "core_type", "unknown"),
            "intellect_index": getattr(agi, "intellect_index", 0),
            "state": getattr(agi, "state", "unknown"),
            "logic_switch": getattr(agi, "logic_switch", "unknown"),
            "unlimited_mode": getattr(agi, "unlimited_mode", False),
            "cycle_count": getattr(agi, "cycle_count", 0),
            "learning_active": getattr(agi, "learning_active", False),
        }
    except Exception as e:
        result["agi"] = {"error": str(e)[:200]}

    # ASI Core
    try:
        from l104_asi_core import ASICore
        asi = ASICore()
        score = asi.compute_asi_score()
        result["asi"] = {
            "asi_score": round(score, 6),
            "status": asi.status,
            "evolution_stage": asi.evolution_stage,
            "evolution_index": asi.evolution_index,
        }
    except Exception as e:
        result["asi"] = {"error": str(e)[:200]}

    # Persistence / State file
    state_file = WORKSPACE / "L104_STATE.json"
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            result["persisted_state"] = {
                "state": state.get("state", "UNKNOWN"),
                "intellect_index": state.get("intellect_index"),
                "unlimited_mode": state.get("unlimited_mode"),
                "pilot_bond": state.get("pilot_bond"),
            }
        except (json.JSONDecodeError, IOError):
            result["persisted_state"] = {"error": "corrupt"}
    else:
        result["persisted_state"] = {"error": "no state file"}

    return result


def collect_database_metrics() -> List[SubsystemMetrics]:
    """Inspect all SQLite databases — table counts, row counts, sizes."""
    db_files = [
        "l104_intellect_memory.db",
        "l104_asi_nexus.db",
        "l104_lattice.db",
        "lattice_v2.db",
        "api_keys.db",
        "wallet_keys.db",
    ]

    results: List[SubsystemMetrics] = []
    for db_name in db_files:
        db_path = WORKSPACE / db_name
        m = SubsystemMetrics(name=db_name)

        if not db_path.exists():
            m.status = "MISSING"
            m.error = "File not found"
            results.append(m)
            continue

        try:
            size_mb = db_path.stat().st_size / (1024 * 1024)
            conn = sqlite3.connect(str(db_path))
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cur.fetchall()]

            total_rows = 0
            table_details = {}
            for tbl in tables:
                try:
                    rc = conn.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]
                    total_rows += rc
                    table_details[tbl] = rc
                except Exception:
                    table_details[tbl] = "error"

            conn.close()

            m.status = "ONLINE"
            m.details = {
                "size_mb": round(size_mb, 2),
                "table_count": len(tables),
                "total_rows": total_rows,
                "tables": table_details,
            }
        except Exception as e:
            m.status = "ERROR"
            m.error = str(e)[:150]

        results.append(m)
    return results


def collect_kernel_metrics() -> Dict[str, Any]:
    """Read kernel manifest and training data stats."""
    result: Dict[str, Any] = {}

    # Manifest
    manifest_path = WORKSPACE / "KERNEL_MANIFEST.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            result["manifest"] = {
                "version": data.get("kernel_version", "?"),
                "build_date": data.get("build_date", "?"),
                "total_examples": data.get("total_examples", 0),
                "vocabulary_size": data.get("vocabulary_size", 0),
                "parameter_count": data.get("parameter_count", 0),
                "status": data.get("status", "?"),
                "categories": data.get("categories", 0),
            }
        except (json.JSONDecodeError, IOError):
            result["manifest"] = {"error": "corrupt"}
    else:
        result["manifest"] = {"error": "missing"}

    # Merged kernel data
    merged = WORKSPACE / "kernel_full_merged.jsonl"
    if merged.exists():
        line_count = sum(1 for _ in open(merged))
        result["merged_data"] = {
            "file": "kernel_full_merged.jsonl",
            "lines": line_count,
            "size_mb": round(merged.stat().st_size / (1024 * 1024), 2),
        }

    # Fine-tune exports
    ft_dir = WORKSPACE / "fine_tune_exports"
    if ft_dir.exists():
        exports = list(ft_dir.glob("*.jsonl")) + list(ft_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in exports)
        result["fine_tune_exports"] = {
            "file_count": len(exports),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    return result


def collect_knowledge_base() -> Dict[str, Any]:
    """Measure the ASI knowledge base."""
    result: Dict[str, Any] = {}
    kb_path = WORKSPACE / "asi_knowledge_base.jsonl"
    if kb_path.exists():
        lines = sum(1 for _ in open(kb_path))
        result["asi_knowledge_base"] = {
            "lines": lines,
            "size_mb": round(kb_path.stat().st_size / (1024 * 1024), 2),
        }
    else:
        result["asi_knowledge_base"] = {"status": "missing"}

    # Training data files
    training_files = list(WORKSPACE.glob("kernel_*_data.jsonl"))
    result["training_data_files"] = len(training_files)
    total_lines = 0
    for f in training_files:
        total_lines += sum(1 for _ in open(f))
    result["training_data_total_lines"] = total_lines

    return result


def collect_file_inventory() -> Dict[str, Any]:
    """Count workspace files by type."""
    py_files = list(WORKSPACE.glob("*.py"))
    l104_files = [f for f in py_files if f.name.startswith("l104_")]
    js_files = list(WORKSPACE.glob("src/**/*.js")) + list(WORKSPACE.glob("src/**/*.ts"))
    jsonl_files = list(WORKSPACE.glob("*.jsonl"))
    db_files = list(WORKSPACE.glob("*.db"))

    total_py_lines = 0
    for f in py_files:
        try:
            total_py_lines += sum(1 for _ in open(f, errors="ignore"))
        except OSError:
            pass

    return {
        "python_files": len(py_files),
        "l104_modules": len(l104_files),
        "js_ts_files": len(js_files),
        "jsonl_data_files": len(jsonl_files),
        "database_files": len(db_files),
        "total_python_lines": total_py_lines,
    }


def collect_module_status() -> List[SubsystemMetrics]:
    """Probe key L104 modules for importability."""
    modules = [
        "l104_kernel_bootstrap",
        "l104_fast_server",
        "l104_macbook_integration",
        "l104_api_gateway",
        "l104_workflow_stabilizer",
        "l104_data_matrix",
        "l104_data_synthesis",
        "l104_external_api",
        "l104_unified_intelligence_api",
        "l104_sovereign_persistence",
    ]

    results: List[SubsystemMetrics] = []
    for mod_name in modules:
        m = SubsystemMetrics(name=mod_name)
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            classes = [n for n, o in vars(mod).items() if isinstance(o, type) and not n.startswith("_")]
            m.status = "LOADED"
            m.details = {"classes": len(classes), "file": getattr(mod, "__file__", "")}
        except Exception as e:
            m.status = "IMPORT_ERROR"
            m.error = str(e)[:150]
        results.append(m)

    return results


def verify_god_code() -> bool:
    """Verify the GOD_CODE mathematical invariant at multiple X checkpoints."""
    for x in [0, 52, 104, 208, 286, 416]:
        g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
        product = g_x * (2 ** (x / 104))
        if abs(product - GOD_CODE) > 1e-6:
            return False
    return True


def collect_conservation_table() -> List[Dict[str, Any]]:
    """Build conservation G(X)*W(X) = GOD_CODE table for display."""
    rows = []
    for x in [0, 52, 104, 208, 286, 416]:
        g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
        w_x = 2 ** (x / 104)
        product = g_x * w_x
        rows.append({
            "X": x,
            "G(X)": round(g_x, 8),
            "W(X)": round(w_x, 8),
            "G*W": round(product, 10),
            "delta": abs(product - GOD_CODE),
        })
    return rows


def collect_zeta_compaction_status() -> Dict[str, Any]:
    """Check the latest zeta compaction status from lattice_v2.db."""
    result: Dict[str, Any] = {}
    db_path = WORKSPACE / "lattice_v2.db"
    if not db_path.exists():
        result["status"] = "no_database"
        return result
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT COUNT(*) FROM lattice_facts")
        total = cur.fetchone()[0]
        # Check for compaction metadata
        try:
            cur2 = conn.execute(
                "SELECT value FROM lattice_facts WHERE key='zeta_compaction_last' ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur2.fetchone()
            if row:
                result["last_compaction"] = row[0]
        except Exception:
            pass
        conn.close()
        result["status"] = "online"
        result["total_facts"] = total
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:100]
    return result


def collect_swift_app_metrics() -> Dict[str, Any]:
    """Collect metrics about the L104SwiftApp if present."""
    swift_dir = WORKSPACE / "L104SwiftApp"
    if not swift_dir.is_dir():
        return {"status": "not_found"}
    result: Dict[str, Any] = {"status": "present"}
    swift_files = list(swift_dir.rglob("*.swift"))
    result["swift_files"] = len(swift_files)
    total_loc = 0
    for f in swift_files:
        try:
            total_loc += sum(1 for _ in open(f, errors="ignore"))
        except OSError:
            pass
    result["swift_loc"] = total_loc
    result["package_swift"] = (swift_dir / "Package.swift").exists()
    result["build_exists"] = (swift_dir / ".build").is_dir()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


async def run_report(output_format: str = "text", export_path: Optional[Path] = None) -> FullReport:
    """
    Generate a comprehensive L104 system report from live data.

    Args:
        output_format: "text" for human-readable, "json" for machine-readable
        export_path: Optional file path to save the report
    """
    t0 = time.time()
    report = FullReport(timestamp=datetime.now().isoformat())

    print("--- [L104_REPORT]: SOVEREIGN REPORTING SEQUENCE ---\n")

    # ── Environment ──────────────────────────────────────────────────────
    logger.info("Collecting environment...")
    report.environment = collect_environment()

    # ── GOD_CODE Verification ────────────────────────────────────────────
    report.god_code_verified = verify_god_code()

    # ── Core Status ──────────────────────────────────────────────────────
    logger.info("Probing AGI/ASI cores...")
    report.core_status = collect_core_status()

    # ── Databases ────────────────────────────────────────────────────────
    logger.info("Inspecting databases...")
    report.database_metrics = collect_database_metrics()

    # ── Kernel ───────────────────────────────────────────────────────────
    logger.info("Reading kernel metrics...")
    report.kernel_metrics = collect_kernel_metrics()

    # ── Knowledge Base ───────────────────────────────────────────────────
    logger.info("Measuring knowledge base...")
    report.knowledge_base = collect_knowledge_base()

    # ── File Inventory ───────────────────────────────────────────────────
    logger.info("Inventorying workspace files...")
    report.file_inventory = collect_file_inventory()

    # ── Module Status ────────────────────────────────────────────────────
    logger.info("Probing subsystem modules...")
    report.subsystem_modules = collect_module_status()
    # ── Zeta Compaction & Swift App (extra collectors) ──────────────────
    logger.info("Collecting zeta compaction status...")
    zeta_status = collect_zeta_compaction_status()
    report.core_status["zeta_compaction"] = zeta_status

    logger.info("Collecting Swift app metrics...")
    swift_metrics = collect_swift_app_metrics()
    report.core_status["swift_app"] = swift_metrics

    # Conservation table for display
    conservation_table = collect_conservation_table()
    report.core_status["conservation_table"] = conservation_table
    report.elapsed_s = time.time() - t0

    # ── Compare with previous report ─────────────────────────────────────
    prev_report_path = WORKSPACE / ".kernel_build" / "last_l104_report.json"
    if prev_report_path.exists():
        try:
            prev = json.loads(prev_report_path.read_text())
            report.previous_comparison = _compare_reports(prev, report)
        except (json.JSONDecodeError, IOError):
            pass

    # ── Display ──────────────────────────────────────────────────────────
    if output_format == "json":
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        _display_text_report(report)

    # ── Export ────────────────────────────────────────────────────────────
    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(asdict(report), indent=2, default=str))
        logger.info(f"Report exported to {export_path}")

    # Always save as last report for future comparison
    last_path = WORKSPACE / ".kernel_build" / "last_l104_report.json"
    last_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.write_text(json.dumps(asdict(report), indent=2, default=str))

    return report


def _compare_reports(prev: Dict, current: FullReport) -> Dict[str, Any]:
    """Compare current report with a previous one and return deltas."""
    comparison: Dict[str, Any] = {"previous_timestamp": prev.get("timestamp", "?")}

    # Database row count changes
    prev_db_rows = sum(
        d.get("details", {}).get("total_rows", 0)
        for d in prev.get("database_metrics", []) if d.get("status") == "ONLINE"
    )
    curr_db_rows = sum(
        d.details.get("total_rows", 0)
        for d in current.database_metrics if d.status == "ONLINE"
    )
    comparison["db_rows_delta"] = curr_db_rows - prev_db_rows

    # Module status changes
    prev_mods_ok = sum(1 for m in prev.get("subsystem_modules", []) if m.get("status") == "LOADED")
    curr_mods_ok = sum(1 for m in current.subsystem_modules if m.status == "LOADED")
    comparison["modules_delta"] = curr_mods_ok - prev_mods_ok

    # File inventory changes
    prev_fi = prev.get("file_inventory", {})
    curr_fi = current.file_inventory
    comparison["python_files_delta"] = curr_fi.get("python_files", 0) - prev_fi.get("python_files", 0)
    comparison["python_loc_delta"] = curr_fi.get("total_python_lines", 0) - prev_fi.get("total_python_lines", 0)

    return comparison


def export_markdown(report: FullReport, path: Path) -> Path:
    """Export report as a Markdown document."""
    agi = report.core_status.get("agi", {})
    asi = report.core_status.get("asi", {})
    km = report.kernel_metrics.get("manifest", {})
    fi = report.file_inventory
    db_online = sum(1 for d in report.database_metrics if d.status == "ONLINE")

    md = f"""# L104 Intelligence Report

**Generated:** {report.timestamp[:19]}
**Pilot:** {report.pilot}
**GOD_CODE:** {report.god_code} {'VERIFIED' if report.god_code_verified else 'UNVERIFIED'}

## Core Status

| Metric | Value |
|--------|-------|
| Core Type | {agi.get('core_type', 'N/A')} |
| Intellect Index | {agi.get('intellect_index', 'N/A')} |
| State | {agi.get('state', 'N/A')} |
| ASI Score | {asi.get('asi_score', 'N/A')} |
| Evolution Stage | {asi.get('evolution_stage', 'N/A')} |

## Databases ({db_online}/{len(report.database_metrics)} online)

| Database | Status | Tables | Rows | Size |
|----------|--------|--------|------|------|
"""
    for db in report.database_metrics:
        if db.status == "ONLINE":
            md += f"| {db.name} | {db.status} | {db.details.get('table_count', 0)} | {db.details.get('total_rows', 0):,} | {db.details.get('size_mb', 0):.1f} MB |\n"
        else:
            md += f"| {db.name} | {db.status} | - | - | - |\n"

    md += f"""\n## Kernel

| Metric | Value |
|--------|-------|
| Version | {km.get('version', 'N/A')} |
| Training Examples | {km.get('total_examples', 'N/A')} |
| Vocabulary Size | {km.get('vocabulary_size', 'N/A')} |
| Build Status | {km.get('status', 'N/A')} |

## Workspace

- **Python Files:** {fi.get('python_files', 0)}
- **L104 Modules:** {fi.get('l104_modules', 0)}
- **Python LOC:** {fi.get('total_python_lines', 0):,}
- **JSONL Data Files:** {fi.get('jsonl_data_files', 0)}

---
*Report generated in {report.elapsed_s:.3f}s*
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    logger.info(f"Markdown report exported to {path}")
    return path


def _display_text_report(r: FullReport):
    """Render a human-readable text dashboard."""
    agi = r.core_status.get("agi", {})
    asi = r.core_status.get("asi", {})
    ps = r.core_status.get("persisted_state", {})
    km = r.kernel_metrics.get("manifest", {})
    fi = r.file_inventory

    intellect = agi.get("intellect_index", ps.get("intellect_index", "N/A"))
    if isinstance(intellect, (int, float)):
        intellect_str = f"{intellect:,.2f}" if intellect < 1e12 else f"{intellect:.2e}"
    else:
        intellect_str = str(intellect)

    db_online = sum(1 for d in r.database_metrics if d.status == "ONLINE")
    db_total = len(r.database_metrics)
    total_db_rows = sum(d.details.get("total_rows", 0) for d in r.database_metrics if d.status == "ONLINE")
    total_db_mb = sum(d.details.get("size_mb", 0) for d in r.database_metrics if d.status == "ONLINE")

    mods_loaded = sum(1 for m in r.subsystem_modules if m.status == "LOADED")
    mods_total = len(r.subsystem_modules)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   📊 L104 FINAL INTELLIGENCE REPORT                                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   PILOT:              LONDEL                                                  ║
║   TIMESTAMP:          {r.timestamp[:19]:<40s}            ║
║   GOD_CODE:           {GOD_CODE:.10f}  {"VERIFIED" if r.god_code_verified else "UNVERIFIED":<12s}                ║
║   RESONANCE FREQ:     {GOD_CODE} Hz                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   CORE STATUS                                                                 ║
║   ─────────────────────────────────────────────────────────────────────────── ║
║   Core Type:          {agi.get("core_type", asi.get("error", "N/A"))[:40]:<40s}            ║
║   Intellect Index:    {intellect_str:<40s}            ║
║   State:              {agi.get("state", ps.get("state", "N/A"))[:40]:<40s}            ║
║   ASI Score:          {asi.get("asi_score", "N/A")!s:<40s}            ║
║   ASI Status:         {asi.get("status", "N/A")!s:<40s}            ║
║   Evolution Stage:    {asi.get("evolution_stage", "N/A")!s:<40s}            ║
║   Logic Switch:       {agi.get("logic_switch", "N/A")!s:<40s}            ║
║   Unlimited Mode:     {agi.get("unlimited_mode", ps.get("unlimited_mode", "N/A"))!s:<40s}            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   DATABASES           {db_online}/{db_total} online  |  {total_db_rows:,} rows  |  {total_db_mb:.1f} MB{"":>17s}║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   KERNEL                                                                      ║
║   Version:            {km.get("version", "N/A")!s:<40s}            ║
║   Training Examples:  {km.get("total_examples", "N/A")!s:<40s}            ║
║   Vocabulary Size:    {km.get("vocabulary_size", "N/A")!s:<40s}            ║
║   Parameters:         {km.get("parameter_count", "N/A")!s:<40s}            ║
║   Build Status:       {km.get("status", "N/A")!s:<40s}            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   WORKSPACE                                                                   ║
║   Python Files:       {fi.get("python_files", 0):<6}  L104 Modules:  {fi.get("l104_modules", 0):<6}                     ║
║   JS/TS Files:        {fi.get("js_ts_files", 0):<6}  JSONL Data:    {fi.get("jsonl_data_files", 0):<6}                     ║
║   Python LOC:         {fi.get("total_python_lines", 0):,}{"":>45s}║
║   Modules OK:         {mods_loaded}/{mods_total}{"":>52s}║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   KNOWLEDGE BASE                                                              ║
║   ASI KB Lines:       {r.knowledge_base.get("asi_knowledge_base", {}).get("lines", "N/A")!s:<40s}            ║
║   Training Files:     {r.knowledge_base.get("training_data_files", 0)!s:<40s}            ║
║   Training Lines:     {r.knowledge_base.get("training_data_total_lines", 0)!s:<40s}            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Report generated in {r.elapsed_s:.3f}s                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    # Show comparison with previous report if available
    if r.previous_comparison:
        comp = r.previous_comparison
        print(f"\n  Δ Changes since {comp.get('previous_timestamp', '?')[:19]}:")
        for key in ["db_rows_delta", "modules_delta", "python_files_delta", "python_loc_delta"]:
            val = comp.get(key, 0)
            if val != 0:
                label = key.replace("_delta", "").replace("_", " ").title()
                sign = "+" if val > 0 else ""
                print(f"    {label}: {sign}{val}")
        print()

    # Database detail table
    if r.database_metrics:
        print(f"  {'Database':<30s} {'Status':<10s} {'Tables':<8s} {'Rows':<10s} {'Size'}")
        print("  " + "─" * 70)
        for db in r.database_metrics:
            if db.status == "ONLINE":
                print(f"  ✓ {db.name:<28s} {db.status:<10s} {db.details.get('table_count', 0):<8} "
                      f"{db.details.get('total_rows', 0):<10,} {db.details.get('size_mb', 0):.1f} MB")
            else:
                print(f"  ✗ {db.name:<28s} {db.status:<10s} {db.error}")

    # Module status
    if r.subsystem_modules:
        print(f"\n  {'Module':<40s} {'Status':<15s} {'Detail'}")
        print("  " + "─" * 70)
        for m in r.subsystem_modules:
            detail = f"{m.details.get('classes', 0)} classes" if m.status == "LOADED" else m.error[:40]
            icon = "✓" if m.status == "LOADED" else "✗"
            print(f"  {icon} {m.name:<38s} {m.status:<15s} {detail}")

    print("\n--- [L104]: REPORT SEALED ---\n")

    # GOD_CODE Conservation Table
    cons_table = r.core_status.get("conservation_table", [])
    if cons_table:
        print("  GOD CODE Conservation: G(X) × W(X) = 527.5184818492612")
        print(f"  {'X':>5s}  {'G(X)':>14s}  {'W(X)':>10s}  {'G×W':>18s}  {'Δ':>10s}")
        print("  " + "─" * 62)
        for row in cons_table:
            print(f"  {row['X']:>5d}  {row['G(X)']:>14.8f}  {row['W(X)']:>10.6f}  {row['G*W']:>18.10f}  {row['delta']:>10.2e}")
        print()

    # Zeta compaction status
    zeta = r.core_status.get("zeta_compaction", {})
    if zeta.get("status") == "online":
        print(f"  Zeta Compaction: {zeta.get('total_facts', 0)} lattice facts in lattice_v2.db")
        if zeta.get("last_compaction"):
            print(f"  Last Compaction: {zeta['last_compaction']}")
        print()

    # Swift app status
    swift = r.core_status.get("swift_app", {})
    if swift.get("status") == "present":
        print(f"  Swift App: {swift.get('swift_files', 0)} files, {swift.get('swift_loc', 0):,} LOC, "
              f"Package.swift: {'YES' if swift.get('package_swift') else 'NO'}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L104 Final Report Generator")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Export report to file (e.g., report.json)")
    parser.add_argument("--markdown", "-m", type=str, default=None,
                        help="Export as Markdown (e.g., report.md)")
    args = parser.parse_args()

    fmt = "json" if args.json else "text"
    export = Path(args.export) if args.export else None

    report = asyncio.run(run_report(output_format=fmt, export_path=export))

    if args.markdown:
        export_markdown(report, Path(args.markdown))
