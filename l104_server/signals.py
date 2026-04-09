# L104 Server — Signal Handlers
# EVO_61: Graceful shutdown and configuration reload

import json
import logging
import os
import signal
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Module-level state
_server_instance: Optional[object] = None
_shutdown_requested: bool = False


def setup_signal_handlers(server_instance: Optional[object] = None) -> None:
    """Setup SIGTERM, SIGHUP, SIGUSR1 handlers for graceful shutdown.

    Args:
        server_instance: uvicorn.Server instance for graceful shutdown
    """
    global _server_instance
    _server_instance = server_instance

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGHUP, _handle_sighup)
    signal.signal(signal.SIGUSR1, _handle_sigusr1)


def _handle_sigterm(signum: int, frame) -> None:
    """Graceful shutdown on SIGTERM (launchd stop / docker stop)."""
    global _shutdown_requested
    if _shutdown_requested:
        return  # Already shutting down
    _shutdown_requested = True
    logger.info(f"[L104] Signal {signum} received — initiating graceful shutdown")
    if _server_instance is not None:
        _server_instance.should_exit = True


def _handle_sighup(signum: int, frame) -> None:
    """SIGHUP: reload configuration without full restart."""
    logger.info("[L104] SIGHUP received — reloading configuration")
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        logger.info("[L104] .env reloaded")
    except Exception as e:
        logger.warning(f"[L104] .env reload failed: {e}")

    try:
        _log_level = os.getenv("LOG_LEVEL", "info").lower()
        _lvl = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(_log_level, logging.INFO)
        logging.getLogger().setLevel(_lvl)
        logger.info(f"[L104] Log level set to {_log_level}")
    except Exception as e:
        logger.warning(f"[L104] Log level reload failed: {e}")


def _handle_sigusr1(signum: int, frame) -> None:
    """SIGUSR1: dump process status to log (health probe from upgrade script)."""
    from config import MAIN_VERSION, MAIN_PIPELINE_EVO

    status = {
        "pid": os.getpid(),
        "version": MAIN_VERSION if 'MAIN_VERSION' in dir() else "61.0.0",
        "pipeline": MAIN_PIPELINE_EVO if 'MAIN_PIPELINE_EVO' in dir() else "EVO_61_SYSTEM_UPGRADE",
        "shutdown_requested": _shutdown_requested,
        "server_alive": _server_instance is not None and not getattr(_server_instance, 'should_exit', True),
    }
    logger.info(f"[L104] SIGUSR1 status dump: {json.dumps(status)}")


def get_shutdown_state() -> bool:
    """Return current shutdown state."""
    return _shutdown_requested


__all__ = [
    "setup_signal_handlers",
    "get_shutdown_state",
]