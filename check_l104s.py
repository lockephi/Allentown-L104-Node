#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 MODULE HEALTH CHECKER — Quick Status Dashboard
═══════════════════════════════════════════════════════════════════════════════

Scans all l104_*.py modules in the workspace, verifies they import cleanly,
catalogues their classes/functions, checks for GOD_CODE alignment, and
produces a concise health dashboard.

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import re
import time
import importlib
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

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [CHECK] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CHECK")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModuleHealth:
    """Health status for a single l104_*.py module."""
    name: str
    filename: str
    importable: bool = False
    import_time_ms: float = 0.0
    error: str = ""
    line_count: int = 0
    size_kb: float = 0.0
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    has_god_code: bool = False
    god_code_value: Optional[float] = None
    god_code_aligned: bool = False
    imports_from: List[str] = field(default_factory=list)  # other l104_* modules this imports


@dataclass
class HealthReport:
    """Aggregate health report across all modules."""
    timestamp: str = ""
    total_modules: int = 0
    importable: int = 0
    failed: int = 0
    total_classes: int = 0
    total_functions: int = 0
    total_lines: int = 0
    total_size_mb: float = 0.0
    god_code_aligned: int = 0
    god_code_misaligned: int = 0
    dependency_edges: int = 0  # total cross-module import links
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    modules: List[ModuleHealth] = field(default_factory=list)
    elapsed_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SCANNER
# ═══════════════════════════════════════════════════════════════════════════════


def discover_l104_files() -> List[Path]:
    """Find all l104_*.py files in the workspace root."""
    return sorted(WORKSPACE.glob("l104_*.py"))


def count_lines(path: Path) -> int:
    """Count lines in a file."""
    try:
        return sum(1 for _ in open(path, errors="ignore"))
    except OSError:
        return 0


def check_god_code_in_source(path: Path) -> Tuple[bool, Optional[float], bool]:
    """
    Scan source for GOD_CODE constant and verify alignment.
    Returns: (has_god_code, value, aligned_with_canonical)
    """
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False, None, False

    pattern = re.compile(r"""GOD_CODE\s*=\s*([\d.]+)""")
    for match in pattern.finditer(text):
        try:
            val = float(match.group(1))
            if abs(val - GOD_CODE) < 100:  # Only consider values near canonical
                aligned = abs(val - GOD_CODE) < 1e-6
                return True, val, aligned
        except ValueError:
            continue
    return False, None, False


def detect_l104_imports(path: Path) -> List[str]:
    """Detect which other l104_* modules this file imports from source text."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []

    imports = set()
    # Match: import l104_xxx  /  from l104_xxx import ...
    import_pattern = re.compile(r"^\s*(?:import|from)\s+(l104_\w+)", re.MULTILINE)
    for match in import_pattern.finditer(text):
        mod_name = match.group(1)
        if mod_name != path.stem:  # don't count self-import
            imports.add(mod_name)
    return sorted(imports)


def probe_module(filepath: Path) -> ModuleHealth:
    """Import and inspect a single l104_*.py module."""
    module_name = filepath.stem
    health = ModuleHealth(
        name=module_name,
        filename=filepath.name,
        line_count=count_lines(filepath),
        size_kb=round(filepath.stat().st_size / 1024, 1),
    )

    # Check GOD_CODE in source (doesn't require import)
    health.has_god_code, health.god_code_value, health.god_code_aligned = \
        check_god_code_in_source(filepath)

    # Detect cross-module dependencies from source
    health.imports_from = detect_l104_imports(filepath)

    # Attempt import
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        health.importable = True
        health.import_time_ms = round((time.time() - t0) * 1000, 1)

        # Catalogue public classes and functions
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            if isinstance(obj, type):
                health.classes.append(name)
            elif callable(obj) and not isinstance(obj, type):
                health.functions.append(name)

    except Exception as e:
        health.importable = False
        health.import_time_ms = round((time.time() - t0) * 1000, 1)
        health.error = str(e)[:150]

    return health


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def check_all(verbose: bool = True, skip_import: bool = False,
              filter_name: Optional[str] = None) -> HealthReport:
    """
    Run health checks on all l104_*.py modules.

    Args:
        verbose: Display per-module details
        skip_import: Only check file-level stats, don't import
        filter_name: If set, only check modules matching this substring
    """
    t0 = time.time()
    report = HealthReport(timestamp=datetime.now().isoformat())

    files = discover_l104_files()
    if filter_name:
        files = [f for f in files if filter_name.lower() in f.stem.lower()]
    report.total_modules = len(files)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🔍 L104 MODULE HEALTH CHECK                                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE:    {GOD_CODE:.10f}                                            ║
║   Modules:     {report.total_modules} discovered                                                 ║
║   Mode:        {"FILE STATS ONLY" if skip_import else "FULL IMPORT + INSPECT":<30s}                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    for filepath in files:
        if skip_import:
            health = ModuleHealth(
                name=filepath.stem,
                filename=filepath.name,
                line_count=count_lines(filepath),
                size_kb=round(filepath.stat().st_size / 1024, 1),
            )
            health.has_god_code, health.god_code_value, health.god_code_aligned = \
                check_god_code_in_source(filepath)
            health.imports_from = detect_l104_imports(filepath)
        else:
            health = probe_module(filepath)

        report.modules.append(health)

        if health.importable:
            report.importable += 1
        elif not skip_import:
            report.failed += 1

        report.total_classes += len(health.classes)
        report.total_functions += len(health.functions)
        report.total_lines += health.line_count
        report.total_size_mb += health.size_kb / 1024

        if health.has_god_code:
            if health.god_code_aligned:
                report.god_code_aligned += 1
            else:
                report.god_code_misaligned += 1

        report.dependency_edges += len(health.imports_from)

    report.total_size_mb = round(report.total_size_mb, 2)
    report.elapsed_s = time.time() - t0

    # Detect circular dependencies
    dep_map = {m.name: set(m.imports_from) for m in report.modules}
    for mod_a, deps_a in dep_map.items():
        for mod_b in deps_a:
            if mod_b in dep_map and mod_a in dep_map[mod_b]:
                pair = tuple(sorted([mod_a, mod_b]))
                if pair not in report.circular_dependencies:
                    report.circular_dependencies.append(pair)

    if verbose:
        _display_dashboard(report, skip_import)

    return report


def _display_dashboard(report: HealthReport, skip_import: bool = False):
    """Pretty-print the health dashboard."""
    # Module table
    if skip_import:
        print(f"  {'Module':<40s} {'Lines':<8} {'Size':<8} {'GOD_CODE'}")
    else:
        print(f"  {'Module':<40s} {'Status':<8} {'ms':<8} {'Lines':<8} {'Classes':<8} {'Funcs':<8} {'GOD_CODE'}")
    print("  " + "─" * 90)

    for m in report.modules:
        gc_str = "✓" if m.god_code_aligned else ("Δ" if m.has_god_code else "—")

        if skip_import:
            print(f"  {m.name:<40s} {m.line_count:<8} {m.size_kb:<7.0f}K {gc_str}")
        else:
            icon = "✓" if m.importable else "✗"
            status = "OK" if m.importable else "FAIL"
            print(f"  {icon} {m.name:<38s} {status:<8s} {m.import_time_ms:<7.0f} "
                  f"{m.line_count:<8} {len(m.classes):<8} {len(m.functions):<8} {gc_str}")
            if not m.importable and m.error:
                print(f"      ↳ {m.error[:80]}")

    # Summary
    if not skip_import:
        health_pct = (report.importable / max(report.total_modules, 1)) * 100
    else:
        # In fast mode, estimate health from GOD_CODE alignment and file existence
        health_pct = (report.god_code_aligned / max(report.total_modules, 1)) * 100 if report.total_modules > 0 else 0

    bar_len = 30
    filled = int(health_pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   📊 SUMMARY                                                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Modules:       {report.total_modules:>4}                                                       ║""")

    if not skip_import:
        print(f"║   Importable:    {report.importable:>4} / {report.total_modules:<4}    [{bar}] {health_pct:.0f}%   ║")
        print(f"║   Failed:        {report.failed:>4}                                                       ║")

    print(f"║   Total Classes: {report.total_classes:>4}     Total Functions: {report.total_functions:>4}                          ║")
    print(f"║   Total Lines:   {report.total_lines:>7,}  Total Size: {report.total_size_mb:.1f} MB                           ║")
    print(f"║   GOD_CODE:      {report.god_code_aligned:>4} aligned  {report.god_code_misaligned:>3} misaligned                            ║")
    print(f"║   Dependencies: {report.dependency_edges:>4} cross-module imports                                  ║")
    print(f"║   Elapsed:       {report.elapsed_s:.3f}s                                                    ║")
    print(f"╚═══════════════════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Top modules by size
    by_lines = sorted(report.modules, key=lambda m: m.line_count, reverse=True)
    print("  Top 10 by line count:")
    for m in by_lines[:10]:
        print(f"    {m.line_count:>7,}  {m.name}")
    print()

    # Dependency graph (top importers)
    with_deps = [m for m in report.modules if m.imports_from]
    if with_deps:
        print("  Module Dependencies:")
        for m in sorted(with_deps, key=lambda x: len(x.imports_from), reverse=True)[:10]:
            deps = ", ".join(m.imports_from[:5])
            extra = f" +{len(m.imports_from) - 5} more" if len(m.imports_from) > 5 else ""
            print(f"    {m.name} → {deps}{extra}")
        print()

    # Most-imported modules (fan-in)
    import_counts: Dict[str, int] = {}
    for m in report.modules:
        for dep in m.imports_from:
            import_counts[dep] = import_counts.get(dep, 0) + 1
    if import_counts:
        print("  Most Imported (fan-in):")
        for mod, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"    {count:>3} imports  {mod}")
        print()

    # Circular dependencies warning
    if report.circular_dependencies:
        print(f"  ⚠ Circular Dependencies ({len(report.circular_dependencies)}):")
        for a, b in report.circular_dependencies:
            print(f"    {a} ⇄ {b}")
        print()

    # Staleness: modules not modified in 30+ days
    stale_threshold = 30 * 86400  # 30 days
    now = time.time()
    stale = []
    for m in report.modules:
        fpath = WORKSPACE / m.filename
        if fpath.exists():
            age = now - fpath.stat().st_mtime
            if age > stale_threshold:
                days = int(age / 86400)
                stale.append((m.name, days))
    if stale:
        print(f"  Stale Modules (>{stale_threshold // 86400} days unchanged):")
        for name, days in sorted(stale, key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {days:>4}d  {name}")
        print()


def export_report(report: HealthReport, path: Optional[Path] = None) -> Path:
    """Export health report to JSON."""
    out = path or (WORKSPACE / ".kernel_build" / "l104_health_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": report.timestamp,
        "total_modules": report.total_modules,
        "importable": report.importable,
        "failed": report.failed,
        "total_classes": report.total_classes,
        "total_functions": report.total_functions,
        "total_lines": report.total_lines,
        "total_size_mb": report.total_size_mb,
        "god_code_aligned": report.god_code_aligned,
        "elapsed_s": round(report.elapsed_s, 3),
        "modules": [asdict(m) for m in report.modules],
    }

    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Report exported to {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L104 Module Health Checker — quick status of all l104_*.py modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python check_l104s.py                # Full import + inspect
  python check_l104s.py --fast         # File stats only (no imports)
  python check_l104s.py --export       # Save report to JSON
  python check_l104s.py --json         # Output as JSON to stdout

GOD_CODE = {GOD_CODE}
        """,
    )
    parser.add_argument("--fast", "-f", action="store_true",
                        help="Skip imports, only check file-level stats")
    parser.add_argument("--export", "-e", action="store_true",
                        help="Export report to JSON file")
    parser.add_argument("--json", action="store_true",
                        help="Print report as JSON to stdout")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only check modules matching this substring")
    args = parser.parse_args()

    report = check_all(verbose=not args.json, skip_import=args.fast,
                       filter_name=args.filter)

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))

    if args.export:
        export_report(report)

    # Exit code: 0 if all importable, 1 if any failures
    sys.exit(0 if report.failed == 0 else 1)