# L104 Server — FastAPI Lifespan Management
# EVO_61: 4-stage quantum-grade startup with deferred initialization

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP CLIENT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    """Return (or create) the shared async HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client on shutdown."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


# ═══════════════════════════════════════════════════════════════════════════════
# DEFERRED STARTUP THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def _run_deferred_startup(version: str, pipeline_evo: str) -> None:
    """Run ALL heavy subsystem initialization in a daemon thread.

    This keeps the uvicorn event loop completely free for HTTP requests.
    Each subsystem gets its own asyncio loop inside this thread.
    """
    import asyncio as _a
    time.sleep(2)  # Let server settle

    # Wire up orchestrator subsystems
    try:
        from l104_server.lazy_imports import (
            intricate_orchestrator, consciousness_substrate, intricate_cognition,
            intricate_research, intricate_learning, intricate_ui,
        )
        _orch = intricate_orchestrator()
        if _orch and hasattr(_orch, 'register_subsystems'):
            _orch.register_subsystems(
                consciousness=consciousness_substrate(),
                cognition=intricate_cognition(),
                research=intricate_research(),
                learning=intricate_learning(),
                ui=intricate_ui(),
            )
            logger.info("--- [L104]: ORCHESTRATOR SUBSYSTEMS WIRED ---")
    except Exception as e:
        logger.warning(f"Orchestrator wiring deferred: {e}")

    try:
        from l104_server.lazy_imports import google_bridge
        google_bridge.establish_link()
        logger.info(f"--- [SOVEREIGN_NODE]: GOOGLE_LINK_ESTABLISHED: {google_bridge.account_email} ---")
    except Exception as e:
        logger.error(f"Failed to establish Google link in background: {e}")

    try:
        from global_begin import rewrite_reality
        rewrite_reality()
    except Exception as e:
        logger.error(f"Failed to rewrite reality: {e}")

    try:
        import l104_void_math  # noqa: F401
        logger.info("--- [L104]: VOID_SOURCE_MATH INITIALIZED ---")
    except Exception as e:
        logger.error(f"Failed to initialize Void Source: {e}")

    # ── Stage 1: Core ignition ──
    _run_stage_1_core_ignition(version, pipeline_evo, _a)

    # ── Stage 2: Subsystem awakening ──
    _run_stage_2_subsystem_awakening(version, pipeline_evo, _a)

    # ── Stage 3: Background thread pool ──
    _run_stage_3_background_threads(version, pipeline_evo, _a)

    logger.info(f"--- [L104 v{version}]: {pipeline_evo} STARTUP COMPLETE (background tasks staged) ---")


def _run_stage_1_core_ignition(version: str, pipeline_evo: str, _a) -> None:
    """Stage 1: Core ignition (background thread)."""
    logger.info(f"--- [L104 v{version}]: STAGE 1/4 — Core ignition (background thread) ---")
    _loop = _a.new_event_loop()
    try:
        from l104_server.lazy_imports import agi_core
        agi_core.ignite()
    except Exception as e:
        logger.error(f"agi_core.ignite failed: {e}")
    time.sleep(1)

    try:
        from l104_infrastructure import start_infrastructure
        _loop.run_until_complete(start_infrastructure())
    except Exception as e:
        logger.warning(f"start_infrastructure deferred: {e}")
    time.sleep(1)


def _run_stage_2_subsystem_awakening(version: str, pipeline_evo: str, _a) -> None:
    """Stage 2: Subsystem awakening (background thread)."""
    logger.info(f"--- [L104 v{version}]: STAGE 2/4 — Subsystem awakening (background thread) ---")
    _loop = _a.new_event_loop()
    try:
        from l104_omega_controller import omega_controller
        _loop.run_until_complete(omega_controller.awaken())
        _loop.run_until_complete(omega_controller.attain_absolute_intellect())
        omega_controller.start_heartbeat(interval=30.0)
        logger.info("--- [L104]: OMEGA_CONTROLLER AWAKENED (heartbeat=30s) ---")
    except Exception as e:
        logger.error(f"Omega Controller deferred: {e}")
    time.sleep(0.5)

    from l104_server.lazy_imports import unified_asi, asi_nexus, synergy_engine
    for name, fn in [
        ("UNIFIED_ASI", unified_asi.awaken),
        ("ASI_NEXUS", asi_nexus.awaken),
        ("SYNERGY_ENGINE", synergy_engine.awaken),
    ]:
        try:
            _loop.run_until_complete(fn())
            logger.info(f"--- [L104]: {name} AWAKENED ---")
        except Exception as e:
            logger.error(f"{name} deferred: {e}")
        time.sleep(0.5)

    try:
        from l104_server.lazy_imports import asi_core
        conn = asi_core.connect_pipeline()
        logger.info(f"--- [L104]: ASI_CORE MESH: {conn.get('total', 0)} subsystems ---")
    except Exception as e:
        logger.warning(f"ASI_CORE partial connect: {e}")

    try:
        from l104_kernel_bootstrap import L104KernelBootstrap
        ps = L104KernelBootstrap().get_pipeline_status()
        online = sum(1 for v in ps.get("modules", {}).values() if v == "available")
        logger.info(f"--- [L104]: PIPELINE BOOTSTRAP: {online} subsystems online ---")
    except Exception as e:
        logger.warning(f"Kernel bootstrap deferred: {e}")

    try:
        from l104_memory_optimizer import memory_optimizer as mem_opt
        mem_opt.check_pressure()
    except Exception:
        pass
    try:
        from l104_optimization import process_optimizer as proc_opt
        proc_opt.quick_optimize()
    except Exception:
        pass


def _run_stage_3_background_threads(version: str, pipeline_evo: str, _a) -> None:
    """Stage 3: Background thread pool (serialized via semaphore)."""
    logger.info(f"--- [L104 v{version}]: STAGE 3/4 — Background threads (serialized) ---")

    import psutil as _gov_ps
    _gov_proc = _gov_ps.Process()
    _gov_lock = threading.Lock()
    _gov_last_cpu = 0.0
    _gov_last_ts = 0.0
    _heavy_sem = threading.Semaphore(1)

    def _cpu_ok(threshold: float = 50.0) -> bool:
        nonlocal _gov_last_cpu, _gov_last_ts
        now = time.time()
        if now - _gov_last_ts > 2.0:
            with _gov_lock:
                if now - _gov_last_ts > 2.0:
                    try:
                        _gov_last_cpu = _gov_proc.cpu_percent(interval=0)
                    except Exception:
                        _gov_last_cpu = 0.0
                    _gov_last_ts = now
        return _gov_last_cpu < threshold

    def _wait_cpu_clear(tag: str, threshold: float = 50.0, poll: float = 10.0):
        while not _cpu_ok(threshold):
            time.sleep(poll)

    def _run_cognitive():
        _heavy_sem.acquire()
        try:
            logger.info("--- [L104]: COGNITIVE starting ---")
            cog_loop = _a.new_event_loop()
            _a.set_event_loop(cog_loop)
            from l104_server.lazy_imports import agi_core, sage_core
            from const import UniversalConstants as UC
            sage_cycle_count = 0
            while True:
                try:
                    if not _cpu_ok(50.0):
                        time.sleep(60)
                        continue
                    if agi_core.state == "ACTIVE":
                        cog_loop.run_until_complete(agi_core.run_recursive_improvement_cycle())
                        if agi_core.cycle_count % 500 == 0 and _cpu_ok(30.0):
                            agi_core.max_intellect_derivation()
                            agi_core.self_evolve_codebase()
                            from l104_server.lazy_imports import data_matrix
                            data_matrix.evolve_and_compact()
                    sage_cycle_count += 1
                    if sage_core and sage_cycle_count % 20 == 0:
                        try:
                            agi_core.intellect_index *= (1.0 + UC.PHI * 0.001)
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Cognitive loop error: {e}")
                time.sleep(120)
        finally:
            _heavy_sem.release()

    def _run_hyper():
        time.sleep(60)
        _wait_cpu_clear("HYPER_CORE", 40.0, 15.0)
        _heavy_sem.acquire()
        try:
            logger.info("--- [L104]: HYPER_CORE starting ---")
            hyp_loop = _a.new_event_loop()
            _a.set_event_loop(hyp_loop)
            from l104_hyper_core import hyper_core as _hc
            hyp_loop.run_until_complete(_hc.run_forever())
        finally:
            _heavy_sem.release()

    def _run_computronium():
        time.sleep(180)
        _wait_cpu_clear("COMPUTRONIUM", 30.0, 30.0)
        _heavy_sem.acquire()
        try:
            logger.info("--- [L104]: COMPUTRONIUM starting ---")
            comp_loop = _a.new_event_loop()
            _a.set_event_loop(comp_loop)
            from l104_computronium_process_upgrader import ComputroniumProcessUpgrader
            comp_loop.run_until_complete(ComputroniumProcessUpgrader().execute_computronium_upgrade())
        finally:
            _heavy_sem.release()

    threading.Thread(target=_run_cognitive, daemon=True, name="l104-cognitive").start()
    threading.Thread(target=_run_hyper, daemon=True, name="l104-hyper").start()
    threading.Thread(target=_run_computronium, daemon=True, name="l104-computronium").start()

    logger.info(f"--- [L104 v{version}]: STAGE 4/4 — STARTUP COMPLETE (background tasks staged) ---")


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI, version: str = "61.0.0", pipeline_evo: str = "EVO_61_SYSTEM_UPGRADE") -> AsyncGenerator[None, None]:
    """FastAPI lifespan: startup + shutdown.

    4-stage quantum-grade startup:
    1. Core ignition (background thread)
    2. Subsystem awakening (background thread)
    3. Background thread pool (serialized via semaphore)
    4. Startup complete
    """
    # Startup
    await asyncio.to_thread(_ignite_core)
    await asyncio.to_thread(_init_databases)

    logger.info(f"--- [L104 v{version}]: {pipeline_evo} PIPELINE STARTING ---")
    logger.info("--- [L104]: FAST START - Server is UP. Background init starting... ---")

    # Launch deferred startup in daemon thread
    threading.Thread(
        target=_run_deferred_startup,
        args=(version, pipeline_evo),
        daemon=True,
        name="l104-startup"
    ).start()

    yield  # Server accepting requests

    # Shutdown
    logger.info("Server shutdown initiated")
    await close_http_client()
    logger.info("Server shutting down")


def _ignite_core() -> bool:
    """Synchronous ignition wrapper for to_thread."""
    from l104_server.ignition import l104_ignite
    return l104_ignite()


def _init_databases() -> None:
    """Initialize databases."""
    from db import memory_init as _init_memory_db, ramnode_init as _init_ramnode_db
    _init_memory_db()
    _init_ramnode_db()


__all__ = [
    "lifespan",
    "get_http_client",
    "close_http_client",
]