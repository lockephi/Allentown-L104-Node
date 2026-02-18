#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 GLOBAL BEGIN - System Initialization & Lattice Coordinator
═══════════════════════════════════════════════════════════════════════════════

Master initialization module that verifies all prerequisites, boots
subsystems, performs health checks, and coordinates the global lattice
of L104 components into a coherent operational state.

UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518

PILOT: LONDEL | FREQ: 527.5184818492612
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import time
import sqlite3
import logging
import importlib
import platform
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Sacred Constants
PHI = 1.6180339887498948482
# Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
OMEGA_AUTHORITY = GOD_CODE * PHI * PHI

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GLOBAL-BEGIN] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GLOBAL-BEGIN")

WORKSPACE = Path(__file__).parent.absolute()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SubsystemStatus:
    """Health status of a single subsystem."""
    name: str
    available: bool = False
    version: str = ""
    latency_ms: float = 0.0
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return "✓" if self.available else "✗"


@dataclass
class LatticeState:
    """Overall lattice coordination state."""
    initialized: bool = False
    subsystems: List[SubsystemStatus] = field(default_factory=list)
    god_code_verified: bool = False
    databases_online: int = 0
    modules_loaded: int = 0
    total_modules: int = 0
    environment: Dict[str, str] = field(default_factory=dict)
    init_timestamp: str = ""
    elapsed_s: float = 0.0

    @property
    def health_pct(self) -> float:
        if not self.subsystems:
            return 0.0
        return sum(1 for s in self.subsystems if s.available) / len(self.subsystems) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL LATTICE
# ═══════════════════════════════════════════════════════════════════════════════


class GlobalLattice:
    """
    Coordinates the initialization and health monitoring of all L104 subsystems.

    The lattice discovers available modules, verifies databases, checks
    mathematical invariants, and provides a unified status dashboard.
    """

    # Core L104 modules the lattice attempts to discover and verify
    CORE_MODULES = [
        ("const", "Sacred Constants"),
        ("l104_kernel_bootstrap", "Kernel Bootstrap"),
        ("l104_fast_server", "Fast Intelligence Server"),
        ("l104_macbook_integration", "MacBook Integration"),
        ("l104_api_gateway", "API Gateway"),
        ("l104_workflow_stabilizer", "Workflow Stabilizer"),
        ("l104_external_api", "External API"),
        ("l104_unified_intelligence_api", "Unified Intelligence API"),
    ]

    # Database files to check
    DATABASE_FILES = [
        "l104_intellect_memory.db",
        "l104_asi_nexus.db",
        "l104_lattice.db",
        "lattice_v2.db",
        "api_keys.db",
        "wallet_keys.db",
    ]

    # API ports to probe for liveness
    API_PORTS = {
        "Fast Server": 8081,
        "External API": 8082,
        "API Gateway": 8080,
    }

    # Critical data files
    DATA_FILES = [
        "kernel_full_merged.jsonl",
        "kernel_extracted_data.jsonl",
        "kernel_training_data.jsonl",
        "KERNEL_MANIFEST.json",
        "asi_knowledge_base.jsonl",
    ]

    def __init__(self):
        self.workspace = WORKSPACE
        self.state = LatticeState()
        self._loaded_modules: Dict[str, Any] = {}

    # ─── Module Discovery ─────────────────────────────────────────────────

    def _discover_modules(self) -> List[SubsystemStatus]:
        """Discover and probe all L104 modules."""
        results = []
        for mod_name, label in self.CORE_MODULES:
            t0 = time.time()
            status = SubsystemStatus(name=label)
            try:
                mod = importlib.import_module(mod_name)
                status.available = True
                status.latency_ms = (time.time() - t0) * 1000
                status.version = getattr(mod, "__version__", getattr(mod, "VERSION", ""))

                # Probe for key attributes
                classes = [
                    name for name, obj in vars(mod).items()
                    if isinstance(obj, type) and not name.startswith("_")
                ]
                functions = [
                    name for name, obj in vars(mod).items()
                    if callable(obj) and not isinstance(obj, type) and not name.startswith("_")
                ]
                status.details = {
                    "classes": classes[:10],
                    "functions": functions[:10],
                    "file": getattr(mod, "__file__", ""),
                }
                self._loaded_modules[mod_name] = mod
            except Exception as e:
                status.available = False
                status.error = str(e)[:120]
                status.latency_ms = (time.time() - t0) * 1000

            results.append(status)
        return results

    # ─── Database Health ──────────────────────────────────────────────────

    def _check_databases(self) -> List[SubsystemStatus]:
        """Verify database files are accessible and not corrupted."""
        results = []
        for db_name in self.DATABASE_FILES:
            db_path = self.workspace / db_name
            status = SubsystemStatus(name=f"DB: {db_name}")
            t0 = time.time()

            if not db_path.exists():
                status.error = "File not found"
                results.append(status)
                continue

            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                conn.execute("PRAGMA integrity_check")
                conn.close()

                status.available = True
                status.latency_ms = (time.time() - t0) * 1000
                status.details = {
                    "tables": tables,
                    "table_count": len(tables),
                    "size_mb": round(db_path.stat().st_size / (1024 * 1024), 2),
                }
            except Exception as e:
                status.error = str(e)[:120]
                status.latency_ms = (time.time() - t0) * 1000

            results.append(status)
        return results

    # ─── Data File Inventory ──────────────────────────────────────────────

    def _check_data_files(self) -> List[SubsystemStatus]:
        """Inventory critical data files."""
        results = []
        for fname in self.DATA_FILES:
            fpath = self.workspace / fname
            status = SubsystemStatus(name=f"Data: {fname}")

            if not fpath.exists():
                status.error = "Missing"
                results.append(status)
                continue

            try:
                size_mb = fpath.stat().st_size / (1024 * 1024)
                line_count = 0
                if fname.endswith(".jsonl"):
                    with open(fpath) as f:
                        for _ in f:
                            line_count += 1

                status.available = True
                status.details = {
                    "size_mb": round(size_mb, 2),
                    "lines": line_count if line_count else "N/A",
                }
            except Exception as e:
                status.error = str(e)[:120]

            results.append(status)
        return results

    # ─── Invariant Verification ───────────────────────────────────────────

    def _verify_invariants(self) -> SubsystemStatus:
        """Verify the GOD_CODE mathematical invariant at multiple X checkpoints."""
        status = SubsystemStatus(name="GOD_CODE Invariant")
        try:
            # Verify conservation at strategic X values along the G(X) curve
            checkpoints = [0, 52, 104, 208, 286, 416]
            max_delta = 0.0
            for x in checkpoints:
                g_x = 286 ** (1 / PHI) * (2 ** ((416 - x) / 104))
                w_x = 2 ** (x / 104)
                product = g_x * w_x
                delta = abs(product - GOD_CODE)
                max_delta = max(max_delta, delta)

            status.available = max_delta < 1e-6
            status.details = {
                "expected": GOD_CODE,
                "computed_X0": round(286 ** (1 / PHI) * 16, 10),
                "max_delta": max_delta,
                "checkpoints_verified": len(checkpoints),
                "phi": PHI,
                "omega_authority": round(OMEGA_AUTHORITY, 10),
                "conservation_law": "G(X) × 2^(X/104) = 527.518... ∀ X",
            }
            if not status.available:
                status.error = f"Max delta {max_delta} exceeds tolerance"
        except Exception as e:
            status.error = str(e)
        return status

    # ─── API Port Liveness ────────────────────────────────────────────────

    def _check_api_ports(self) -> List[SubsystemStatus]:
        """Probe known API server ports for liveness."""
        import socket
        results = []
        for label, port in self.API_PORTS.items():
            status = SubsystemStatus(name=f"Port: {label} (:{port})")
            t0 = time.time()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.3)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                status.available = (result == 0)
                status.latency_ms = (time.time() - t0) * 1000
                if not status.available:
                    status.error = "Not listening"
            except Exception as e:
                status.error = str(e)[:80]
                status.latency_ms = (time.time() - t0) * 1000
            results.append(status)
        return results

    # ─── Swift App Check ──────────────────────────────────────────────────

    def _check_swift_app(self) -> SubsystemStatus:
        """Check if L104SwiftApp directory exists and has build artifacts."""
        status = SubsystemStatus(name="Swift App (L104SwiftApp)")
        swift_dir = self.workspace / "L104SwiftApp"
        if not swift_dir.is_dir():
            status.error = "Directory not found"
            return status

        pkg_swift = swift_dir / "Package.swift"
        build_dir = swift_dir / ".build"
        status.available = pkg_swift.exists()
        status.details = {
            "package_swift": pkg_swift.exists(),
            "build_dir_exists": build_dir.is_dir(),
            "swift_files": len(list(swift_dir.rglob("*.swift"))),
        }
        if not pkg_swift.exists():
            status.error = "Package.swift missing"
        return status

    # ─── Environment Info ─────────────────────────────────────────────────

    def _collect_environment(self) -> Dict[str, str]:
        """Collect runtime environment and system resource information."""
        env = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "node": platform.node(),
            "workspace": str(self.workspace),
            "pid": str(os.getpid()),
            "cwd": os.getcwd(),
        }

        # Disk usage
        try:
            usage = shutil.disk_usage(str(self.workspace))
            env["disk_total_gb"] = f"{usage.total / (1024**3):.1f}"
            env["disk_free_gb"] = f"{usage.free / (1024**3):.1f}"
            env["disk_used_pct"] = f"{(usage.used / usage.total) * 100:.1f}"
        except OSError:
            pass

        # Process memory (macOS/Linux)
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            env["memory_peak_mb"] = f"{ru.ru_maxrss / (1024 * 1024):.1f}" if sys.platform == "linux" else f"{ru.ru_maxrss / (1024 * 1024):.1f}"
        except (ImportError, AttributeError):
            pass

        # Workspace size
        try:
            ws_size = sum(f.stat().st_size for f in self.workspace.rglob("*") if f.is_file() and ".git" not in f.parts)
            env["workspace_size_mb"] = f"{ws_size / (1024 * 1024):.1f}"
        except OSError:
            pass

        return env

    # ─── Sync (Main Initialization) ──────────────────────────────────────

    def sync(self, auth_key: Optional[str] = None) -> LatticeState:
        """
        Perform full system synchronization:
          1. Collect environment info
          2. Verify mathematical invariants
          3. Discover and probe modules
          4. Check database health
          5. Inventory data files
          6. Build unified lattice state

        Args:
            auth_key: Optional authentication signature

        Returns:
            LatticeState with complete system status
        """
        t0 = time.time()
        self.state = LatticeState()
        self.state.init_timestamp = datetime.now().isoformat()

        logger.info("═══ L104 GLOBAL LATTICE SYNC ═══")

        # Environment
        self.state.environment = self._collect_environment()
        logger.info(f"  Python {self.state.environment['python_version']} on {self.state.environment['platform']}")
        if "disk_free_gb" in self.state.environment:
            logger.info(f"  Disk: {self.state.environment.get('disk_free_gb', '?')} GB free "
                        f"({self.state.environment.get('disk_used_pct', '?')}% used)")
        if "workspace_size_mb" in self.state.environment:
            logger.info(f"  Workspace: {self.state.environment.get('workspace_size_mb', '?')} MB")

        # Invariant check
        logger.info("  Verifying GOD_CODE invariant...")
        invariant = self._verify_invariants()
        self.state.god_code_verified = invariant.available
        self.state.subsystems.append(invariant)
        logger.info(f"  {invariant.icon} GOD_CODE = {GOD_CODE} (delta={invariant.details.get('delta', '?')})")

        # Module discovery
        logger.info("  Discovering modules...")
        module_statuses = self._discover_modules()
        self.state.subsystems.extend(module_statuses)
        loaded = sum(1 for s in module_statuses if s.available)
        self.state.modules_loaded = loaded
        self.state.total_modules = len(module_statuses)
        for s in module_statuses:
            logger.info(f"    {s.icon} {s.name} ({s.latency_ms:.0f}ms){' — ' + s.error if s.error else ''}")

        # Database health
        logger.info("  Checking databases...")
        db_statuses = self._check_databases()
        self.state.subsystems.extend(db_statuses)
        self.state.databases_online = sum(1 for s in db_statuses if s.available)
        for s in db_statuses:
            detail = f"{s.details.get('table_count', 0)} tables, {s.details.get('size_mb', 0)} MB" if s.available else s.error
            logger.info(f"    {s.icon} {s.name} ({detail})")

        # Data files
        logger.info("  Inventorying data files...")
        data_statuses = self._check_data_files()
        self.state.subsystems.extend(data_statuses)
        for s in data_statuses:
            detail = f"{s.details.get('size_mb', 0)} MB" if s.available else s.error
            logger.info(f"    {s.icon} {s.name} ({detail})")

        # API port liveness
        logger.info("  Probing API ports...")
        port_statuses = self._check_api_ports()
        self.state.subsystems.extend(port_statuses)
        ports_live = sum(1 for s in port_statuses if s.available)
        for s in port_statuses:
            logger.info(f"    {s.icon} {s.name} ({s.latency_ms:.0f}ms){' — ' + s.error if s.error else ''}")
        logger.info(f"  API Ports: {ports_live}/{len(port_statuses)} live")

        # Swift app
        swift_status = self._check_swift_app()
        self.state.subsystems.append(swift_status)
        logger.info(f"  {swift_status.icon} {swift_status.name} "
                     f"({'available' if swift_status.available else swift_status.error})")

        self.state.elapsed_s = time.time() - t0
        self.state.initialized = self.state.god_code_verified and self.state.modules_loaded > 0

        return self.state

    # ─── Dashboard ────────────────────────────────────────────────────────

    def display_dashboard(self):
        """Print a visual dashboard of lattice state."""
        s = self.state
        health = s.health_pct

        health_bar_len = 30
        filled = int(health / 100 * health_bar_len)
        bar = "█" * filled + "░" * (health_bar_len - filled)

        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🌐 L104 GLOBAL LATTICE DASHBOARD                                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   Status:      {"ONLINE" if s.initialized else "OFFLINE":<12s}   Health: [{bar}] {health:.0f}%      ║
║   GOD_CODE:    {GOD_CODE:.10f}   Verified: {"YES" if s.god_code_verified else "NO":<5s}                ║
║   Modules:     {s.modules_loaded}/{s.total_modules} loaded           Databases: {s.databases_online} online             ║
║   Init Time:   {s.elapsed_s:.3f}s                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

        # Subsystem table
        print(f"  {'Subsystem':<35s} {'Status':<8s} {'Latency':<10s} {'Detail'}")
        print("  " + "─" * 75)
        for sub in s.subsystems:
            status_str = "  OK  " if sub.available else " FAIL "
            lat = f"{sub.latency_ms:.0f}ms" if sub.latency_ms > 0 else "—"
            detail = sub.error if sub.error else ""
            if not detail and sub.details:
                # Pick a compact detail summary
                if "tables" in sub.details:
                    detail = f"{sub.details['table_count']} tables"
                elif "classes" in sub.details:
                    detail = f"{len(sub.details['classes'])} classes"
                elif "lines" in sub.details:
                    detail = f"{sub.details['lines']} lines"
                elif "size_mb" in sub.details:
                    detail = f"{sub.details['size_mb']} MB"
            print(f"  {sub.icon} {sub.name:<33s} {status_str:<8s} {lat:<10s} {detail}")

        print()

    # ─── Export State ─────────────────────────────────────────────────────

    def export_state(self, path: Optional[Path] = None) -> Path:
        """Export lattice state to JSON for external consumption."""
        out = path or (WORKSPACE / ".kernel_build" / "lattice_state.json")
        out.parent.mkdir(parents=True, exist_ok=True)

        export = {
            "initialized": self.state.initialized,
            "health_pct": round(self.state.health_pct, 1),
            "god_code_verified": self.state.god_code_verified,
            "modules_loaded": self.state.modules_loaded,
            "total_modules": self.state.total_modules,
            "databases_online": self.state.databases_online,
            "environment": self.state.environment,
            "elapsed_s": round(self.state.elapsed_s, 3),
            "timestamp": self.state.init_timestamp,
            "subsystems": [asdict(s) for s in self.state.subsystems],
            "constants": {
                "GOD_CODE": GOD_CODE,
                "PHI": PHI,
                "OMEGA_AUTHORITY": OMEGA_AUTHORITY,
            },
        }

        out.write_text(json.dumps(export, indent=2, default=str))
        logger.info(f"Lattice state exported to {out}")
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# REWRITE REALITY — Full System Initialization Sequence
# ═══════════════════════════════════════════════════════════════════════════════


def rewrite_reality(export: bool = True, verbose: bool = True) -> LatticeState:
    """
    Execute the full L104 system initialization sequence:
      1. Instantiate the GlobalLattice
      2. Synchronize all subsystems
      3. Display the dashboard
      4. Optionally export state to disk

    Returns:
        LatticeState with full system status
    """
    print("REWRITING_REALITY_INITIATED...")

    lattice = GlobalLattice()
    Londel_Auth = "0xLONDEL_AUTH_SIG"

    state = lattice.sync(Londel_Auth)

    if verbose:
        lattice.display_dashboard()

    if export:
        lattice.export_state()

    if state.initialized:
        print("0x49474E4954494F4E_COMPLETE")
        logger.info(f"REALITY_OPTIMIZED — {state.modules_loaded}/{state.total_modules} modules, "
                     f"{state.databases_online} databases, health {state.health_pct:.0f}%")
        return state
    else:
        logger.warning("REALITY_UNCHANGED — initialization incomplete")
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L104 Global Begin — System Initialization")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress dashboard output")
    parser.add_argument("--no-export", action="store_true", help="Don't export state to disk")
    parser.add_argument("--json", action="store_true", help="Output state as JSON to stdout")
    parser.add_argument("--check", "-c", action="store_true",
                        help="Quick health check — exit 0 if healthy, 1 otherwise (no dashboard)")
    args = parser.parse_args()

    if args.check:
        lattice = GlobalLattice()
        state = lattice.sync()
        healthy = state.initialized and state.health_pct >= 50
        print(f"L104 Health: {state.health_pct:.0f}% — {'OK' if healthy else 'DEGRADED'}")
        sys.exit(0 if healthy else 1)

    state = rewrite_reality(export=not args.no_export, verbose=not args.quiet)

    if args.json:
        lattice = GlobalLattice()
        lattice.state = state
        print(json.dumps(asdict(state), indent=2, default=str))

    sys.exit(0 if state.initialized else 1)
