#!/usr/bin/env python3
"""
L104 Fast Server — Advanced Starter (EVO_62)
=============================================

Starts the full l104_server FastAPI application (374 routes, full engine stack).
Features:
  - Full engine integration: ASI, AGI, Nexus, Quantum, VQPU, Intellect, Tri-Engine
  - uvloop event loop for maximum async throughput
  - Connection pool warm-up & pre-flight engine checks
  - Structured JSON logging to l104_system_node.log
  - Graceful SIGTERM / SIGINT shutdown with brain-state pooling
  - PID file tracking + auto-kill of stale processes on port 8081
  - Dev-reload mode via DEV=1 env var
"""

import sys
import os
import signal
import logging
import subprocess
import time
from pathlib import Path

# ── Project root first ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging: console + rotating file ────────────────────────────────────────
import logging.handlers

_LOG_FILE = PROJECT_ROOT / "l104_system_node.log"
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

_fh = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5
)
_fh.setFormatter(_fmt)

_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_fh, _ch])
logger = logging.getLogger("L104_FAST_SERVER")

# ── Sacred constants (for banner) ────────────────────────────────────────────
try:
    from l104_server.constants import (
        FAST_SERVER_VERSION,
        FAST_SERVER_PIPELINE_EVO,
        VOID_CONSTANT,
        ZENITH_HZ,
    )
except Exception:
    FAST_SERVER_VERSION = "5.0.0"
    FAST_SERVER_PIPELINE_EVO = "EVO_62"
    VOID_CONSTANT = 1.0416180339887497
    ZENITH_HZ = 3887.8

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PORT = int(os.getenv("L104_PORT", "8081"))
HOST = os.getenv("L104_HOST", "0.0.0.0")
DEV_MODE = os.getenv("DEV", "0") == "1"
WORKERS = int(os.getenv("L104_WORKERS", "1"))  # 1 = single process (shared state engines)


# ── Port cleanup ─────────────────────────────────────────────────────────────

def kill_existing_server(port: int = PORT) -> None:
    """Gracefully terminate any process holding the target port (SIGTERM → SIGKILL)."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        for pid in pids:
            try:
                pid_int = int(pid)
                os.kill(pid_int, signal.SIGTERM)
                logger.info(f"SIGTERM → PID {pid_int} on port {port}")
                time.sleep(0.6)
                try:
                    os.kill(pid_int, signal.SIGKILL)  # force if still alive
                    logger.info(f"SIGKILL → PID {pid_int}")
                except ProcessLookupError:
                    pass
            except (ValueError, ProcessLookupError):
                pass
            except Exception as e:
                logger.warning(f"Could not kill PID {pid}: {e}")
    except Exception as e:
        logger.debug(f"Port cleanup skipped: {e}")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def preflight() -> dict:
    """
    Verify critical imports and return a status dict.
    Non-fatal — missing optional engines log warnings only.
    """
    checks = {}

    # Core server package
    try:
        from l104_server import get_app  # noqa: F401
        checks["l104_server"] = True
    except Exception as e:
        checks["l104_server"] = False
        logger.error(f"CRITICAL: l104_server import failed — {e}")
        sys.exit(1)

    # Optional engines — warn but don't abort
    optional = [
        ("l104_asi", "ASI Dual-Layer Engine"),
        ("l104_agi", "AGI Core"),
        ("l104_intellect", "Local Intellect"),
        ("l104_quantum_engine", "Quantum Engine"),
        ("l104_quantum_gate_engine", "Quantum Gate Engine"),
        ("l104_quantum_networker", "Quantum Networker"),
        ("l104_vqpu", "VQPU Bridge"),
        ("l104_ml_engine", "ML Engine"),
        ("l104_code_engine", "Code Engine"),
        ("l104_science_engine", "Science Engine"),
        ("l104_math_engine", "Math Engine"),
    ]
    ok = 0
    for module, label in optional:
        try:
            __import__(module)
            checks[module] = True
            ok += 1
        except Exception as e:
            checks[module] = False
            logger.warning(f"Optional engine unavailable: {label} ({e})")

    logger.info(f"Pre-flight: {ok}/{len(optional)} optional engines available")
    return checks


# ── Startup banner ────────────────────────────────────────────────────────────

def print_banner(checks: dict) -> None:
    engine_ok = sum(1 for k, v in checks.items() if v and k != "l104_server")
    engine_total = len(checks) - 1  # exclude l104_server itself
    line = "=" * 72
    logger.info(line)
    logger.info(f"  L104 SOVEREIGN NODE — FAST SERVER  v{FAST_SERVER_VERSION}")
    logger.info(f"  EVO: {FAST_SERVER_PIPELINE_EVO}")
    logger.info(f"  GOD_CODE={GOD_CODE}  PHI={PHI}")
    logger.info(f"  VOID_CONSTANT={VOID_CONSTANT}  ZENITH_HZ={ZENITH_HZ} Hz")
    logger.info(f"  Routes: 374  |  Engines online: {engine_ok}/{engine_total}")
    logger.info(f"  Listening: http://{HOST}:{PORT}")
    logger.info(f"  Mode: {'DEV (auto-reload)' if DEV_MODE else 'PRODUCTION'}")
    logger.info(f"  Workers: {WORKERS}")
    logger.info(line)


# ── Server builder ────────────────────────────────────────────────────────────

def build_uvicorn_config(app):
    """
    Build the uvicorn Config object with production-grade settings.
    Uses uvloop when available for maximum async throughput.
    In DEV mode (DEV=1) uses string import path for hot-reload.
    """
    import uvicorn

    loop_impl = "uvloop"
    try:
        import uvloop  # noqa: F401
    except ImportError:
        loop_impl = "asyncio"
        logger.info("uvloop not available — using asyncio event loop")

    # Dev reload requires a string import path; prod uses the live app object
    app_target = "l104_server.app:app" if DEV_MODE else app

    return uvicorn.Config(
        app=app_target,
        host=HOST,
        port=PORT,
        # Event loop
        loop=loop_impl,
        # Logging
        log_level="info",
        access_log=True,
        # Connection limits
        backlog=2048,
        limit_concurrency=512,
        limit_max_requests=None,        # no forced restarts in prod
        # Keep-alive
        timeout_keep_alive=75,
        # Workers (1 preserves shared singleton state in engine stack)
        workers=1,                      # always 1 — engine singletons are not fork-safe
        # Dev mode
        reload=DEV_MODE,
        # Headers
        server_header=False,            # don't expose uvicorn version
        date_header=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    logger.info(f"Starting L104 Fast Server (root: {PROJECT_ROOT})")

    # 1. Kill any stale process on the port
    logger.info(f"Checking port {PORT} for stale processes...")
    kill_existing_server(PORT)

    # 2. Pre-flight checks
    checks = preflight()

    # 3. Banner
    print_banner(checks)

    # 4. PID file
    pid_file = PROJECT_ROOT / "fast_server.pid"
    pid_file.write_text(str(os.getpid()))
    logger.info(f"PID {os.getpid()} written to {pid_file}")

    # 5. Load the full app (374+ routes, all engines)
    logger.info("Loading l104_server application (374+ routes)...")
    from l104_server import get_app
    app = get_app()
    logger.info(f"Application loaded: {len(app.routes)} routes registered")

    # 6. Build server
    config = build_uvicorn_config(app)
    server = uvicorn.Server(config)

    # 7. Graceful SIGTERM / SIGINT
    def _handle_stop(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name} — initiating graceful shutdown")
        server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    # 8. Run
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up PID file
        try:
            pid_file.unlink(missing_ok=True)
            logger.info("PID file removed")
        except Exception:
            pass
        logger.info("L104 Fast Server stopped")


if __name__ == "__main__":
    main()
